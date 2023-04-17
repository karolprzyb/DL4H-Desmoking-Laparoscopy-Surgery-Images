import os
import logging
import copy
from typing import Tuple, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
                              
from utils import *
from models import UNet
from losses import PSNR
from SynthTrainer import SynthTrainer
import arg_parser
import json

import math
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


def gamma_embedding(gammas, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param gammas: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """

class EmbedSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channel, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channel, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channel
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of embedding channels.
    :param dropout: the rate of dropout.
    :param out_channel: if specified, the number of out channels.
    :param use_conv: if True and out_channel is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channel=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            nn.Conv2d(channels, self.out_channel, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channel),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
            ),
        )

        if self.out_channel == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channel, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class UNet_Transformer(nn.Module):
    """
    The full UNet model with attention and embedding.
    :param in_channel: channels in the input Tensor, for image colorization : Y_channels + X_channels .
    :param inner_channel: base channel count for the model.
    :param out_channel: channels in the output Tensor.
    :param res_blocks: number of residual blocks per downsample.
    :param attn_res: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mults: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channel,
        inner_channel,
        out_channel,
        res_blocks,
        attn_res,
        dropout=0,
        channel_mults=(1, 2, 4, 8),
        conv_resample=True,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    ):

        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.out_channel = out_channel
        self.res_blocks = res_blocks
        self.attn_res = attn_res
        self.dropout = dropout
        self.channel_mults = channel_mults
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        cond_embed_dim = inner_channel * 4
        self.cond_embed = nn.Sequential(
            nn.Linear(inner_channel, cond_embed_dim),
            SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        ch = input_ch = int(channel_mults[0] * inner_channel)
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(mult * inner_channel),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * inner_channel)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, out_channel=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(inner_channel * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(inner_channel * mult)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channel=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
        )

    def forward(self, x, gammas):
        """
        Apply the model to an input batch.
        :param x: an [N x 2 x ...] Tensor of inputs (B&W)
        :param gammas: a 1-D batch of gammas.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        gammas = gammas.view(-1, )
        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))

        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


def prepareModel(args:arg_parser.argparse.Namespace, num_images:int, dataset_tuple:Tuple) \
                            -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    
    # Smoke detector
    if args.load_net_smoke_detect:
        # -- Load arguments
        smoke_args_path = os.path.join( os.path.dirname(args.load_net_smoke_detect), "commandline_args.txt")
        if not os.path.isfile(smoke_args_path):
            raise ValueError('Could not find commandline_args.txt file in the same directory as {}'.format(args.load_net_smoke_detect))
        smoke_args = arg_parser.argparse.Namespace()
        with open(smoke_args_path) as smoke_cmd_file:
            smoke_args_dict = json.load(smoke_cmd_file)
            smoke_args.__dict__.update(smoke_args_dict)
        # -- Check if args from load_net_smoke_detect has the same single_frame setting as here
        if  smoke_args.single_frame != args.single_frame:
            raise ValueError('Smoke network needs to be {} but the arg is: {} '.format(args.single_frame, 
                                                                                       smoke_args.single_frame))
        # -- Load smoke detect network
        smoke_detect_net = UNet(output_channels = num_images,
                                input_channels = 3*num_images,
                                layer_num_channels = smoke_args.layers, 
                                drop_out_layers_dec = smoke_args.drop_out,
                                drop_out_layers_enc = [0 for _ in range(len(smoke_args.layers))])
        smoke_detect_net.load_state_dict(torch.load(args.load_net_smoke_detect))
        smoke_detect_net.eval()
        for param in smoke_detect_net.parameters():
            param.requires_grad = False
    else:
        smoke_detect_net = None

    in_channel = num_images*3 + 3 
    diff_model = diffusion_net(UNet_Transformer(image_size=256,
                                                in_channel=in_channel,
                                                out_channel=dataset_tuple[-1].shape[0],
                                                inner_channel=64,
                                                channel_mults=[1,2,4,8],
                                                attn_res=[16],
                                                num_head_channels=32,
                                                res_blocks=2,
                                                dropout=0.2))

    return diff_model, smoke_detect_net


class diffusion_net(torch.nn.Module):
    def __init__(self, 
                 net:torch.nn.Module,
                 variance_train:torch.Tensor  = torch.linspace(10**-6, 0.01, 2000), 
                 variance_refine:torch.Tensor = torch.linspace(10**-4, 0.09, 100)):
        super().__init__()
        # Internally the notation is based on:
        #       https://www.overleaf.com/read/pmdmzdngzjjj
        #       except \overline{alpha} is switched for gamma
        self.denoise_net = net

        # Type casting for MyPy
        self.beta_train:torch.Tensor
        self.beta_refine:torch.Tensor
        self.alpha_train:torch.Tensor
        self.alpha_refine:torch.Tensor
        self.gamma_train:torch.Tensor
        self.gamma_refine:torch.Tensor
        self.gamma_refine_prev:torch.Tensor
        self.sigma_2_train:torch.Tensor
        self.sigma_2_refine:torch.Tensor

        # Save beta
        self.register_buffer('beta_train', variance_train)
        self.register_buffer('beta_refine', variance_refine)

        # Save alpha
        self.register_buffer('alpha_train', 1 - self.beta_train)
        self.register_buffer('alpha_refine',1 - self.beta_refine)

        # Save gamma
        self.register_buffer('gamma_train',  torch.cumprod(self.alpha_train, dim=0))
        self.register_buffer('gamma_refine', torch.cumprod(self.alpha_refine, dim=0))
        self.register_buffer('gamma_refine_prev', torch.cat([torch.tensor([1]).to(self.gamma_refine.device), 
                                                            self.gamma_refine[:-1]]))
    
        # Save sigma^2
        self.register_buffer('sigma_2_refine', self.beta_refine \
                                                * (1 - self.gamma_refine_prev) \
                                                / (1 - self.gamma_refine))

    def setVarScheduleRefine(self, variance_refine:torch.Tensor) -> None:
        self.beta_refine = variance_refine
        self.alpha_refine = 1 - self.beta_refine
        self.gamma_refine = torch.cumprod(self.alpha_refine, dim=0)
        self.gamma_refine_prev = torch.cat([torch.tensor([1]).to(self.gamma_refine.device), 
                                            self.gamma_refine[:-1]])
        self.sigma_2_refine = self.beta_refine * (1 - self.gamma_refine_prev) / (1 - self.gamma_refine)

    @torch.no_grad()
    def generate_image(self, conditioned_image:torch.Tensor, mask:torch.Tensor = None) \
            -> torch.Tensor:
        '''
            Given a conditioned image, will run through the reverse model to generate an ouptut image
                 x_{t-1} = 1/sqrt(alpha_t) * ( x_t  - net(x_t, gamma_t) *beta_t / sqrt(1 - gamma_t) ) + z
            where z ~ N(0, sigma^2)
        '''
        x = torch.randn_like(conditioned_image)

        for t in reversed(range(0, len(self.gamma_refine))):

            # Estimate 0 directly from this time-stamp
            denoise_output = self.denoise_net(torch.cat([conditioned_image, x], dim=1),
                                              self.gamma_refine[t].view((x.shape[0], 1)))
            x_0 = torch.sqrt(1. / self.gamma_refine[t] ) * x \
                    - torch.sqrt( 1. / self.gamma_refine[t] - 1) * denoise_output

            # Clip result so it's always within bounds (helps with output I think...)
            x_0.clamp_(-1., 1.)
            
            # Do this magic
            coeff1 = self.beta_refine[t] * torch.sqrt(self.gamma_refine_prev[t]) / (1. - self.gamma_refine[t])
            coeff2 = (1. - self.gamma_refine_prev[t]) * torch.sqrt(self.alpha_refine[t]) / (1. - self.gamma_refine[t])
            x = coeff1 * x_0 + coeff2 * x
            
            # Add noise
            noise = torch.sqrt(self.sigma_2_refine[t]) * torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = x + noise

            # Apply mask if it exists
            if mask is not None:
                x = mask*x + conditioned_image*(1 - mask)

        return x

    def forward(self, x_0:torch.Tensor, conditioned_image:torch.Tensor, mask:torch.Tensor = None)\
                         -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Computes the diffusion model denoised and noise for a sample x_0
                net( sqrt(gamma) x_0 * epsilon + sqrt(1 - gamma) * epsilon, gamma), epsilon
            where epsilon ~ N(0, I) and gamma is sampled from self.gamma_train
        '''
        # Pick random timestep
        t = torch.randint(1, len(self.gamma_train), (x_0.shape[0],)).to(x_0.device)

        # Pick gamma/variance of noise at the timesteps
        gamma_t   = torch.gather(self.gamma_train, 0, t).unsqueeze(-1)
        gamma_t_1 = torch.gather(self.gamma_train, 0, t-1).unsqueeze(-1)
        gamma = (gamma_t - gamma_t_1) * torch.rand_like(gamma_t) + gamma_t_1 
        
        # Get noise and predicted image at the timestep
        epsilon = torch.randn_like(x_0)
        x = torch.sqrt(gamma.view(-1, 1, 1, 1)) * x_0 + torch.sqrt(1 - gamma.view(-1, 1, 1, 1)) * epsilon

        if mask is not None:
            x = mask*x + conditioned_image*(1 - mask)

        pred_epsilon = self.denoise_net(torch.cat([conditioned_image, x], dim=1), gamma)

        # Apply mask if its inputted
        if mask is not None:
            pred_epsilon = mask*pred_epsilon
            epsilon = mask*epsilon

        # return noise prediction and noise 
        return pred_epsilon, epsilon
    

class DiffTrainer(SynthTrainer):
    def __init__(self, scheduler:lr_scheduler.LinearLR, net_EMA:diffusion_net, ema_weight:float,
                *args, **kwargs):
        '''
            Optimizes net_EMA
        '''
        self.scheduler = scheduler
        self.net_EMA = net_EMA
        self.ema_weight = ema_weight
        super().__init__(*args, **kwargs)

    def trainIteration(self, net:diffusion_net, dataset_tuple:Tuple, net_smoke:torch.nn.Module = None):
        smoke_img, true_mask, bg_img = dataset_tuple
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)
        bg_img = bg_img.to(device=self.device, dtype=torch.float32)

        if self.single_frame:
            smoke_img = smoke_img[:, :3, :, :]

        # Detect smoke from smoke detector network if it exists
        if net_smoke:
            mask_pred = (net_smoke(smoke_img) + 1)/2.0
        else:
            mask_pred = None

        # Run inference
        denoise, noise = net(bg_img, smoke_img, mask = mask_pred)

        # Compute loss and back-propogate
        loss = self.training_criterion(denoise, noise)

        if torch.isnan(loss).any():
            logging.warning("Loss has NaN!!!!")
            return

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Update EMA network
        for current_params, ma_params in zip(net.parameters(), self.net_EMA.parameters()):
            ma_params.data = self.ema_weight * ma_params.data + (1 - self.ema_weight) * current_params

        self.writer.add_scalar('training loss', loss, self.global_step)

    def validationIteration(self, net:diffusion_net, dataset_tuple:Tuple, net_smoke:torch.nn.Module = None):
        smoke_img, true_mask, bg_img = dataset_tuple
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)
        bg_img = bg_img.to(device=self.device, dtype=torch.float32)

        if self.single_frame:
            smoke_img = smoke_img[:, :3, :, :]

        # Detect smoke from smoke detector network if it exists
        if net_smoke:
            mask_pred = (net_smoke(smoke_img) + 1)/2.0
        else:
            mask_pred = None

        # Run inference but with the EMA!!!
        bg_pred = self.net_EMA.generate_image(smoke_img, mask = mask_pred)

        # Get loss
        loss = self.validation_criterion(bg_pred, bg_img)
        if not torch.isnan(loss).any():
            self.total_validation_loss += loss.item()

        # Save images to show
        if self.validation_step in self.val_indices_to_show:
            bg_img =  ( (bg_img[0].cpu() + 1) / 2)
            smoke_img = ( (smoke_img[0].cpu() + 1) / 2)
            true_mask = ( (true_mask[0].cpu() + 1) / 2 )
            bg_pred = ( (bg_pred[0].cpu() + 1) / 2 )
            
            # Note: Moved image tensor to CPU to save gpu memory.
            self.bg_img_list.append(transforms.Resize(int(bg_img.shape[-1]))(bg_img))
            self.smoke_img_list.append(transforms.Resize(int(smoke_img.shape[-1]))(smoke_img[:3, : ,:]))
            self.true_mask_list.append(transforms.Resize(int(true_mask.shape[-1]))(true_mask[0,  : ,:].unsqueeze(0)))
            self.bg_pred_list.append(transforms.Resize(int(bg_pred.shape[-1]))(bg_pred))
            if mask_pred is not None:
                mask_pred = mask_pred[0].cpu()
                self.mask_pred_list.append(transforms.Resize(int(mask_pred.shape[-1]))(mask_pred[0,  : ,:].unsqueeze(0)))

    def testIteration(self, net:diffusion_net, dataset_tuple:Tuple, net_smoke:torch.nn.Module = None):
        smoke_img, _, _ = dataset_tuple

        if self.single_frame:
            smoke_img = smoke_img[:, :3, :, :]

        # Set device for images
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)

        # Detect smoke from smoke detector network if it exists
        if net_smoke:
            mask_pred = (net_smoke(smoke_img) + 1)/2.0
        else:
            mask_pred = None

        # Run inference
        bg_pred = self.net_EMA.generate_image(smoke_img, mask = mask_pred)

        # Save images to show 
        smoke_img = transforms.CenterCrop((bg_pred.shape[-2], bg_pred.shape[-1]))(smoke_img)
        smoke_img = ( (smoke_img[0].cpu() + 1) / 2)
        bg_pred   = ( (bg_pred[0].cpu()   + 1) / 2)
        

        self.smoke_img_list.append(transforms.Resize(int(smoke_img.shape[-1]))(smoke_img[:3, : ,:]))
        self.bg_pred_list.append(transforms.Resize(int(bg_pred.shape[-1]))(bg_pred))
        if mask_pred is not None:
            mask_pred = mask_pred[0].cpu()
            self.mask_pred_list.append(transforms.Resize(int(mask_pred.shape[-1]))(mask_pred[0,  : ,:].unsqueeze(0)))

    def save_network(self, net:torch.nn.Module, path:str):
        super().save_network(net, path)
        path_for_ema = os.path.splitext(path)[0] + "_ema" + ".pth"
        torch.save(self.net_EMA.state_dict(), path_for_ema)


if __name__ == '__main__':
    # Command line arguments
    parser = arg_parser.create()
    parser.add_argument('--ema', type=int, metavar='ema_weight', default=0.9999)
    parser.add_argument('--load_u_net', type=str, metavar='load_file_for_u_net_in_diffusion_model', 
                        default='')
    parser.add_argument('--load_net_smoke_detect', type=str, metavar='path to *.pth holding saved smoke network' + \
                                                                   ' and in the same directory as commandline_args.txt. ' + \
                                                                    'This net should be traijned by smoke_detector.py')

     # --- Parse args and save
    args = parser.parse_args()
    args.single_frame = True # HARD-CODED SINCE MULTI-FRAME HASN'T BEEN FINISHED YET
    args.__dict__["run_file"] = os.path.abspath(__file__)
    arg_parser.save(args)

    # Save Tensorboard
    writer = SummaryWriter(args.save)

    # Save log output in file-path too
    logging.basicConfig(filename=os.path.join(args.save, 'run.log'),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    # Pick device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        device = torch.device('cpu')

    # Prepare datasets
    from synth_data_settings import train_dataset, val_dataset, real_smoke_dataset, multi_frame
    
    # Prepare model
    if args.single_frame:
        num_images = 1
    else:
        num_images = len(multi_frame) + 1
    net, net_smoke = prepareModel(args, num_images, train_dataset[0])
    net = net.to(device)
    if net_smoke:
        net_smoke = net_smoke.to(device)
    logging.info(net)
    training_criterion = torch.nn.L1Loss()
    validation_criterion = PSNR(2)

    # Load saved network
    if args.load_net:
        net.load_state_dict(torch.load(args.load_net))
    
    if args.load_u_net:
        data = torch.load(args.load_u_net)
        # Remove keys not relevant to denoise_fn
        data = {k.split('denoise_fn.',1)[1]: data[k] for k in data if k.startswith('denoise_fn')}
        net.denoise_net.load_state_dict(data)

    # Initialize optimizer
    optimizer = Adam(net.parameters(), lr=args.lr, betas=args.adam) 
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor = 1e-8, total_iters = int(1024/args.batch)*10**4)
    net_EMA = copy.deepcopy(net)
    net_EMA.eval()

    # Initialize and run trainer
    trainer = DiffTrainer(scheduler = scheduler,
                          net_EMA = net_EMA,
                          ema_weight = args.ema,
                          device = device, 
                          training_criterion = training_criterion, 
                          validation_criterion = validation_criterion, 
                          writer = writer, 
                          optimizer = optimizer,
                          save_path = args.save, num_epoch = args.epochs, batch_size = args.batch,
                          single_frame = args.single_frame,
                          run_val_and_test_every_steps = 5000
                          )
    trainer.train(net, train_dataset, val_dataset, real_smoke_dataset, net_smoke = net_smoke)
