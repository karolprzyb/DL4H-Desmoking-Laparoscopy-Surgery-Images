import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import CenterCrop
from torchvision.ops import DeformConv2d

from typing import List, Tuple, Union

def getOutputSize_conv2d(input_size:torch.Size, out_channels:int, padding:int=0, 
                            dilation:int=1, kernel_size:int=4, stride:int=1):
    '''
        Formula from PyTorch that computes the output size of a conv2d layer:
            https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    
    '''
    get_res = lambda res:int((res + 2*padding - dilation * (kernel_size - 1) - 1)/stride + 1)
    return torch.Size((input_size[0], out_channels, get_res(input_size[2]), get_res(input_size[3])))


class ConvRelu(nn.Module):
    '''
        Conv + Relu module:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: kernel size of convolution layers
            num_conv: number of conv + relu layers
            drop_out: probability of drop_out,  Note: 0 implies no drop_out layer
            leaky_relu: Use leaky relu or regular relu layers
            leaky_relu_slope: Negative slope of leaky relu
            batch_norm: adds batch norm at the end of the layers
            stride: Stride in conv layers
            padding: Padding in conv layers
            crop_output: Crops the output resolution to the input resolution. Useful when padding
            initialize_weights: initializes weight of conv layers with he distribution and BN
                                with gaussian of std of 0.02
    '''
    def __init__(self, in_channels:int, out_channels:int, 
                kernel_size:int = 4,
                dilation:int = 1,
                num_conv:int = 1,
                drop_out:float = 0.0,
                leaky_relu:bool = False,
                leaky_relu_slope:float = 0.2,
                batch_norm:bool = True,
                stride:int = 1,
                padding:int = 0,
                crop_output:bool=False,
                initialize_weights = True):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_conv = num_conv
        self.stride = stride
        self.padding = padding
        self.crop_output = crop_output

        for layer in range(num_conv):
            # Input convolution
            if layer == 0:
                self.conv_relu = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                         dilation=dilation, stride=stride, padding=padding))
            # Regular convolution layer
            else:
                self.conv_relu.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                                        dilation=dilation, stride=stride, padding=padding))

            # Initialize convolution weights with he distribution since the activation is relu
            if initialize_weights:
                if leaky_relu:
                    torch.nn.init.kaiming_uniform_(self.conv_relu[-1].weight, a=leaky_relu_slope, 
                                                    mode='fan_in', nonlinearity='leaky_relu')
                else:
                    torch.nn.init.kaiming_uniform_(self.conv_relu[-1].weight, 
                                                    mode='fan_in', nonlinearity='relu')

            # Optional batch norm
            if batch_norm:
                self.conv_relu.append(nn.BatchNorm2d(out_channels))
                if initialize_weights:
                    torch.nn.init.constant_(self.conv_relu[-1].bias.data, 0.0)
                    nn.init.normal_(self.conv_relu[-1].weight.data, 1.0, 0.02)

            # relu layer
            if leaky_relu:
                self.conv_relu.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
            else:
                self.conv_relu.append(nn.ReLU())
        
        if drop_out != 0:
            self.conv_relu.append(nn.Dropout(drop_out))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.crop_output:
            return CenterCrop((x.shape[-2], x.shape[-1]))(self.conv_relu(x))
        else:
            return self.conv_relu(x)
    
    def _resolution_computation(self, input_size:torch.Size, num_conv:int) -> torch.Size:
        '''
            If crop_out, output resolution is same as input (assumes sufficient padding)
            If not crop_out, computes resolution from here: 
                https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        '''
        if self.crop_output:
            return torch.Size((input_size[0], self.out_channels, input_size[2], input_size[3]))

        out_size = getOutputSize_conv2d(input_size, self.out_channels, self.padding,
                                        self.dilation, self.kernel_size, self.stride)

        # termination condition for recursive computation
        if num_conv == 1:
            return out_size
        # recursive call to iterate over all the convolutions
        else:
            return self._resolution_computation(out_size, num_conv-1)

    def getOutputSize(self, input_size:torch.Size) -> torch.Size:
        '''
            Returns the size of output given an input size
                Input: input_size of tuple size (B, C, H, W)
                Output: output_size of tuple size (B, C_out, H_out, W_out)
        '''
        return self._resolution_computation(input_size, self.num_conv)


class ResBlock(nn.Module):
    '''
        ResBlock Module:
            channels: number of channels
    '''
    def __init__(self, channels:int):
        super().__init__()
        self.res_block = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, 
                                                    stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(channels, channels, kernel_size=3, 
                                                    stride=1, padding=1))

    def forward(self, x):
        return x + self.res_block(x)

    def getOutputSize(self, input_size:torch.Size) -> torch.Size:
        '''
            Returns the size of output given an input size
                Input: input_size of tuple size (B, C, H, W)
                Output: output_size of tuple size (B, C_out, H_out, W_out)
        '''
        return input_size


class Down(nn.Module):
    """
        Encoder layer with single convolution and maxpooling
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: kernel size of convolution layers
            num_conv: number of conv + relu layers
            drop_out: probability of drop_out,  Note: 0 implies no drop_out layer
            leaky_relu: Use leaky relu or regular relu layers
            batch_norm: adds batch norm at the end of the layers
            padding: Padding in conv layers
            use_max_pool: Use max pool to downsample or will set stride=2 for convrelu.
    """
    def __init__(self, in_channels:int, out_channels:int, 
                kernel_size:int = 4,
                num_conv:int = 1,
                drop_out:float = 0.0,
                leaky_relu:bool = True,
                batch_norm:bool = True,
                padding:int = 2,
                use_max_pool:bool = False):
        super().__init__()
        self.use_max_pool = use_max_pool
        if self.use_max_pool:
            self.conv = nn.Sequential(ConvRelu(in_channels=in_channels, out_channels=out_channels, 
                                                kernel_size=kernel_size, num_conv=num_conv, 
                                                drop_out=drop_out, leaky_relu=leaky_relu, 
                                                batch_norm=batch_norm, padding=padding),
                                      nn.MaxPool2d(2))
        else:
            self.conv = nn.Sequential(ConvRelu(in_channels=in_channels, out_channels=out_channels, 
                                                kernel_size=kernel_size, num_conv=num_conv, 
                                                drop_out=drop_out, leaky_relu=leaky_relu, 
                                                batch_norm=batch_norm, padding=padding, stride=2))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
            Output:
                x: goes through convrelu and max pool if set
        '''
        return self.conv(x)

    def getOutputSize(self, input_size:torch.Size) -> torch.Size:
        '''
            Returns the size of output given an input size
                Input: input_size of tuple size (B, C, H, W)
                Output: output_size of tuple size (B, C_out, H_out/2, W_out/2), (B, C_out, H_out, W_out)
        '''
        conv_relu_size = self.conv[0].getOutputSize(input_size)
        if self.use_max_pool:
            return torch.Size([input_size[0], conv_relu_size[1], 
                                int(conv_relu_size[2]/2), int(conv_relu_size[3]/2)])
        else:
            return conv_relu_size


class Up(nn.Module):
    """
        Decoder layer with single convolution
            in_channels: number of input channels
            out_channels: number of output channels after up_sampling
            kernel_size: kernel size of convolution layers
            num_conv: number of conv + relu layers
            drop_out: probability of drop_out,  Note: 0 implies no drop_out layer
            leaky_relu: Use leaky relu or regular relu layers
            batch_norm: adds batch norm at the end of the layers
            padding: Padding in additional (i.e. more than the conv2dtranpsose) conv-relu layers
            initialize_weights: initializes weight of conv layers with he distribution
    """

    def __init__(self, in_channels:int, out_channels:int, 
                kernel_size:int = 4,
                num_conv:int = 1,
                drop_out:float = 0.0,
                leaky_relu:bool = False,
                batch_norm:bool = True,
                padding:int = 2,
                initialize_weights:bool = True):
        super().__init__()
        self.out_channels = out_channels

        # Conv relu before convtranspose2d
        self.conv_relu = nn.Sequential()
        self.num_conv = num_conv
        if num_conv > 1:
            self.conv_relu.append(ConvRelu(in_channels=in_channels, 
                                            out_channels=in_channels, 
                                            kernel_size=kernel_size, num_conv = num_conv - 1, 
                                            drop_out=0, leaky_relu=leaky_relu, leaky_relu_slope=0.2,
                                            batch_norm=batch_norm, padding=padding,
                                            initialize_weights = initialize_weights))

        # conv transpose 2d operation to up-sample              
        self.conv_relu.append(nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size=kernel_size, stride=2))
        # Initialize convolution weights with he distribution since the activation is relu
        if initialize_weights:
            if leaky_relu:
                torch.nn.init.kaiming_uniform_(self.conv_relu[-1].weight, a=0.2, 
                                                mode='fan_in', nonlinearity='leaky_relu')
            else:
                torch.nn.init.kaiming_uniform_(self.conv_relu[-1].weight, 
                                                mode='fan_in', nonlinearity='relu')

        # Optional batch norm
        if batch_norm:
            self.conv_relu.append(nn.BatchNorm2d(out_channels))
            if initialize_weights:
                torch.nn.init.constant_(self.conv_relu[-1].bias.data, 0.0)
                nn.init.normal_(self.conv_relu[-1].weight.data, 1.0, 0.02)

        # relu layer
        if leaky_relu:
            self.conv_relu.append(nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv_relu.append(nn.ReLU())
        
        if drop_out != 0:
            self.conv_relu.append(nn.Dropout(drop_out))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Note output is cropped to input shape since conv2dtranspose increases the resolution.
        return CenterCrop((2*x.shape[-2], 2*x.shape[-1]))(self.conv_relu(x))

    def getOutputSize(self, input_size:torch.Size) -> torch.Size:
        '''
            Returns the size of output given an input size
                Input: input_size of tuple size (B, C, H, W)
                Output: output_size of tuple size (B, C_out, H_out, W_out)
        '''
        if self.num_conv > 1:
            input_size = self.conv_relu[0].getOutputSize(input_size)
        return torch.Size([input_size[0], self.out_channels, 
                            int(input_size[2]*2), int(input_size[3]*2)])


class Encoder(nn.Module):
    """
        Encoder module using Conv-Relu Layers:
            input_channels: number of channels for input image
            output_channels: number of channels for output encoded layer
            layer_num_channels: number of channels in the internal layers
            drop_out_layers: drop out probability for each layer. Note: zero implies no dropout
                         Should be same length of layer_num_channels
                         Last convolution layer will NOT have drop-outs
            save_skips: If true, will save all the features of every layer for encoder
                        Set to true for U-Net style architectures. They are saved in:
                            self.skip_connection_0, ..., self.skip_connection_{N-1} 
            kernel_size: kernel size of convolution layers
            leaky_relu: Use leaky relu or regular relu layers
            batch_norm: adds batch norm to each layer. Note that input layer always has no BN
            padded: If true, adds padding such that the resolution is always exactly doubled or halfed each layer
            use_max_pool: use max pooling for encoding (otherwise uses conv stride)
    """

    def __init__(self,
                input_channels:int=3,
                output_channels:int=512,
                layer_num_channels:List[int] = [64, 128, 256, 512, 512, 512],
                drop_out_layers:List[float] = [0, 0, 0, 0, 0, 0],
                drop_out_in_last_layer:bool = False,
                save_skips:bool = True,
                kernel_size:int = 4, # I think the padding doesn't work for kernel = 3.
                leaky_relu:bool = True,
                batch_norm:bool = True,
                use_max_pool:bool = False):
        super().__init__()

        self.save_skips = save_skips
        if use_max_pool:
            padding = int(np.ceil((kernel_size - 1)/2.0))
        else:
            padding = int((kernel_size / 2.0) - 1)

        # Input layer/convolution
        self.input_layer = ConvRelu(in_channels = input_channels, 
                                    out_channels = layer_num_channels[0],
                                    kernel_size = kernel_size,
                                    batch_norm = False,
                                    padding = int(np.ceil((kernel_size - 1)/2.0)),
                                    drop_out = 0.0,
                                    crop_output = True)

        # Appending encoder layers
        self.down_layers = nn.ModuleList([])
        for channel_idx, num_channels in enumerate(layer_num_channels):
            # Initialize buffer for skip connections
            if self.save_skips:
                self.register_buffer("skip_connection_{}".format(channel_idx), torch.empty((1)), persistent=False)

            # Handle case for last layer
            if channel_idx == len(layer_num_channels) - 1:
                out_channels = output_channels
            else:
                out_channels = layer_num_channels[channel_idx + 1]

            self.down_layers.append(Down(in_channels=num_channels, 
                                         out_channels=out_channels, 
                                         kernel_size=kernel_size,
                                         num_conv=1, 
                                         drop_out=drop_out_layers[channel_idx], 
                                         leaky_relu=leaky_relu, 
                                         batch_norm=batch_norm,
                                         padding=padding,
                                         use_max_pool=use_max_pool))

        # output layer
        self.last_layer = ConvRelu(in_channels = output_channels, 
                                    out_channels = output_channels,
                                    kernel_size = kernel_size,
                                    padding = int(np.ceil((kernel_size - 1)/2.0)),
                                    drop_out = 0.5 if drop_out_in_last_layer else 0, 
                                    crop_output = True) # Crop output if to keep resolution
                                    # through this layer since padding may increase resolution

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        x = self.input_layer(x)
        
        for layer_idx, down_layer in enumerate(self.down_layers):
            if self.save_skips:
                setattr(self, "skip_connection_{}".format(layer_idx), x)
            x = down_layer(x)

        x = self.last_layer(x)

        return x

    def _resolution_computation(self, input_size:torch.Size, num_down:int) -> torch.Size:
        '''
            Recursively computes the resolution. 
        '''
        if num_down == 0:
            return self.last_layer.getOutputSize(input_size)
        else:
            return self._resolution_computation(self.down_layers[-num_down].getOutputSize(input_size), num_down-1)

    def getOutputSize(self, input_size:torch.Size) -> torch.Size:
        '''
            Returns the size of output given an input size for the forward pass
            See getOutputSizeSkip() for skip output sizes
                Input: input_size of tuple size (B, C, H, W)
                Output: output_size of tuple size (B, C_out, H_out, W_out)
        '''
        return self._resolution_computation(self.input_layer.getOutputSize(input_size), len(self.down_layers))

    def _resolution_computation_skip(self, input_size:torch.Size, skip:int, current_layer:int = 0) -> torch.Size:
        '''
            Recursively computes the resolution of skip connection
        '''
        if skip == current_layer:
            return input_size
        else:
            return self._resolution_computation_skip(self.down_layers[current_layer].getOutputSize(input_size),
                                                     skip, current_layer + 1 )

    def getOutputSizeSkip(self, input_size:torch.Size, skip_num:int) -> torch.Size:
        '''
            Returns the size of output skip connection given an input size for the forward pass
                Input: input_size of tuple size (B, C, H, W)
                       skip_num is the index of the skip 0,...,N-1 for N skips
                Output: output_size of tuple size (B, C_out, H_out, W_out)
        '''
        if not self.save_skips:
            return torch.Size([1])
        else:
            return self._resolution_computation_skip(self.input_layer.getOutputSize(input_size), skip_num)


class Decoder(nn.Module):
    """
        Decoder module using Conv-Relu Layers:
            input_channels: number of channels for input encoded image
            output_channels: number of channels for output image
            layer_num_channels: number of channels in the internal layers
            drop_out_layers: drop out probability for each layer. Note: zero implies no dropout
                         Should be same length of layer_num_channels
            skip_layer_inputs: If true, will multiply the number of channels per layer EXCEPT the first layer
                        and allow skip inputs to the forward call
                        This is useful when doing a U-Net style architecture
            skip_layer_mult: multiplier for input channels if using skip_layer_inputs
            kernel_size: kernel size of convolution layers
            leaky_relu: Use leaky relu or regular relu layers
            batch_norm: adds batch norm to each layer. Note that output layers always has BN
    """

    def __init__(self,
                input_channels:int=512,
                output_channels:int=3, 
                layer_num_channels:List[int] = [512, 512, 512, 256, 128, 64],
                drop_out_layers:List[float] = [0.5, 0.5, 0.5, 0, 0, 0],
                skip_layer_inputs:bool = True,
                skip_layer_mult:int=2,
                kernel_size:int = 4,
                leaky_relu:bool = False,
                batch_norm:bool = True):
        super().__init__()

        self.skip_layer_inputs = skip_layer_inputs
        if not skip_layer_inputs:
            skip_layer_mult = 1
        self.skip_layer_mult = skip_layer_mult
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.layer_num_channels = layer_num_channels

        # Input layer
        self.input_layer = Up(in_channels=input_channels, 
                              out_channels=layer_num_channels[0],
                              kernel_size=kernel_size, 
                              num_conv=1, 
                              drop_out=drop_out_layers[0], 
                              leaky_relu=leaky_relu, 
                              batch_norm=batch_norm)

        self.up_layers = nn.ModuleList([])
        for channel_idx, num_channels in enumerate(layer_num_channels[:-1]):
            # Append decoder layer
            self.up_layers.append(Up(in_channels=num_channels*skip_layer_mult, 
                                     out_channels=layer_num_channels[channel_idx + 1],
                                     kernel_size=kernel_size, 
                                     num_conv=1, 
                                     drop_out=drop_out_layers[channel_idx + 1], 
                                     leaky_relu=leaky_relu, 
                                     batch_norm=batch_norm))

        # Output layer
        self.output_layer = torch.nn.Sequential(ConvRelu(in_channels = layer_num_channels[-1]*skip_layer_mult, 
                                                         out_channels = layer_num_channels[-1],
                                                         kernel_size = kernel_size,
                                                         padding = int(np.ceil((kernel_size - 1)/2.0)),
                                                         drop_out = 0, 
                                                         crop_output = True),
                                                nn.Conv2d(layer_num_channels[-1],
                                                          output_channels, 1),
                                                nn.Tanh()
                                                )


    def forward(self, x:torch.Tensor, skip_input:List[torch.Tensor] = []) -> torch.Tensor:
        # Input layer
        x = self.input_layer(x)

        # Decoder layers
        for layer_idx, up_layer in enumerate(self.up_layers):
            # Skip connection with crop
            if self.skip_layer_inputs and len(skip_input):
                x = torch.concat((CenterCrop((x.shape[-2], x.shape[-1])) (skip_input[layer_idx]), x), dim=1)
            x = up_layer(x)
        
        # Last layer
        if self.skip_layer_inputs and len(skip_input):
            x = torch.concat((CenterCrop((x.shape[-2], x.shape[-1])) (skip_input[-1]), x), dim=1)
        x = self.output_layer(x)

        # Do output layer
        return x

    def _resolution_computation(self, input_size:torch.Size, num_up:int) -> torch.Size:
        '''
            Recursively computes the resolution. 
        '''
        if self.skip_layer_inputs:
            input_size = torch.Size((input_size[0], self.skip_layer_mult*input_size[1], 
                                        input_size[2], input_size[3]))
        if num_up == 0:
            output_size =  self.output_layer[0].getOutputSize(input_size)
            return torch.Size((output_size[0], self.output_channels, output_size[2], output_size[3]))
        else:
            return self._resolution_computation(self.up_layers[-num_up].getOutputSize(input_size), num_up-1)

    def getOutputSize(self, input_size:torch.Size) -> torch.Size:
        '''
            Returns the size of output given an input size for the forward pass
                Input: input_size of tuple size (B, C, H, W)
                Output: output_size of tuple size (B, C_out, H_out, W_out)
        '''
        return self._resolution_computation(self.input_layer.getOutputSize(input_size), len(self.up_layers))


class UNet(nn.Module):
    """
    UNet implementation:
        input_channels: number of channels for input image
        output_channels: number of output channels
        layer_num_channels: number of channels in the internal layers (will be mirrored for encoder/decoder)
        drop_out_layers_enc: drop out probability for encoder layers. Note: zero implies no dropout
                         Should be the same length as layer_num_channels
        drop_out_layers_dec: drop out probability for decoder layers. Note: zero implies no dropout
                         Should be the same length as layer_num_channels
        kernel_size: kernel size of convolution layers
        num_conv: number of conv + relu layers in each encoder/decoder layer
        leaky_relu_encoder: Use leaky relu or regular relu layers in encoder
        leaky_relu_decoder: Use leaky relu or regular relu layers in dencoder
        batch_norm: adds batch norm to each layer. Note that input and output layers always have no BN
    """
    def __init__(self,
                input_channels:int=3, 
                output_channels:int=1, 
                layer_num_channels:List[int] = [64, 128, 256, 512, 512, 512],
                drop_out_layers_enc:List[float] = [0, 0, 0, 0, 0, 0],
                drop_out_layers_dec:List[float] = [0.5, 0, 0, 0, 0, 0],
                drop_out_in_mid_layer:bool = False,
                kernel_size:int = 4,
                leaky_relu_encoder:bool = True,
                leaky_relu_decoder:bool = False,
                batch_norm:bool = True):
        super().__init__()
        self.num_layers = len(layer_num_channels)
        self.encoder = Encoder(input_channels = input_channels,
                               output_channels = layer_num_channels[-1],
                               layer_num_channels = layer_num_channels[:-1],
                               drop_out_layers = drop_out_layers_enc,
                               drop_out_in_last_layer=drop_out_in_mid_layer,
                               save_skips = True,
                               kernel_size = kernel_size,
                               leaky_relu = leaky_relu_encoder,
                               batch_norm = batch_norm)
        layer_num_channels.reverse()
        self.decoder = Decoder(input_channels = layer_num_channels[0],
                               output_channels = output_channels,
                               layer_num_channels = layer_num_channels[1:],
                               drop_out_layers = drop_out_layers_dec,
                               skip_layer_inputs = True,
                               kernel_size = kernel_size,
                               leaky_relu = leaky_relu_decoder,
                               batch_norm = batch_norm)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x, [getattr(self.encoder, "skip_connection_{}".format(layer_idx)) 
                             for layer_idx in reversed(range(self.num_layers - 1))])
        return x

    def getOutputSize(self, input_size:torch.Size) -> torch.Size:
        return self.decoder.getOutputSize(self.encoder.getOutputSize(input_size))


class DefBlock(nn.Module):
    '''
        Deformable block which contains an offset convolution and the deformable convolution

            channels: number of channels for input to deformable convolution
            offset_channels: number of channels for input to compute offsets
            deform_groups: number of deformable groups
            kernel_size: kernel size used for offset convolution and deformable convolution
            padding: padding used
            bias: enables bias to offset convolutions
    '''
    def __init__(self, channels:int,
                offset_channels:int,
                deform_groups:int = 8,
                kernel_size:int = 3,
                padding:int = 1,
                bias:bool = True):
        super().__init__()
        self.offset_conv = nn.Conv2d(offset_channels, deform_groups * 2 * kernel_size**2,
                                     kernel_size = kernel_size, padding = padding, bias = bias)
        self.deform = DeformConv2d(channels, channels, kernel_size = kernel_size, 
                                    padding = padding, groups = deform_groups)

    def forward(self, x:torch.Tensor, x_offset:torch.Tensor) -> torch.Tensor:
        '''
            Applies deformable convolution to x given features from x_offset:
                x [B x C x H x W] input to be deformed by deform_conv 
                x_offset [B x C_offset x H x W] input to offset_conv which 
                    computes offsets for deform_conv
        
        '''
        return self.deform(x, self.offset_conv(x_offset))

    def getOutputSize(self, input_size:torch.Size) -> torch.Size:
        '''
            Returns the size of output given an input size
                Input: input_size of tuple size (B, C, H, W)
                Output: output_size of tuple size (B, C_out, H_out, W_out)
        '''
        return input_size


class AlignNet(nn.Module):
    '''
        Alignment Network from TDAN: 
        https://openaccess.thecvf.com/content_CVPR_2020/papers/Tian_TDAN_Temporally-Deformable_Alignment_Network_for_Video_Super-Resolution_CVPR_2020_paper.pdf

            input_channels: number of channels for input image
            output_channels: number of channels for output image
            hidden_channels: the number of hidden channels for the feature space
            num_feat_extraction_blocks: number of feature extraction blocks before the deformable layers
            num_def_blocks: number of deformable blocks
            bias: enable bias to convolution layers (or not)

        TODO: Add option for alignment net from here which should be minor modifications...
            https://openaccess.thecvf.com/content/CVPR2022/papers/Dudhane_Burst_Image_Restoration_and_Enhancement_CVPR_2022_paper.pdf
            https://github.com/akshaydudhane16/BIPNet/blob/b0384d5a80f6c5040ba476551c97183cf1fd35a3/Burst%20De-noising/Network.py
    '''
    def __init__(self, input_channels:int, 
                output_channels:int,
                hidden_channels:int = 64,
                num_feat_extraction_blocks:int = 5,
                bias:bool = True):
        super().__init__()

        # Default settings
        deform_groups = 8
        kernel_size = 3
        padding = 1

        # Save for forward pass
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Input layers for feature extraction
        self.conv_first = nn.Conv2d(input_channels, hidden_channels, 
                                    kernel_size = kernel_size, padding = padding, bias = bias)
        self.relu = nn.ReLU(inplace=True)
        self.residual_layer = nn.Sequential(*[ResBlock(channels=hidden_channels) 
                                                for _ in range(num_feat_extraction_blocks)])

        # Bottle neck
        self.bottleneck = nn.Conv2d(hidden_channels*2, hidden_channels, 
                                    kernel_size = kernel_size, padding = padding, bias = bias)

        # Deformable layers.
        # NOTE: layer 1,2, 4 are regular deformable layers applies to the features
        #       in both the current image and the image to be aligned.
        #       According to the authors, this enhancing the transformation 
        #       flexibility and capability of the module
        self.def_1 = DefBlock(channels = hidden_channels, offset_channels = hidden_channels,
                                deform_groups = deform_groups, kernel_size = kernel_size,
                                padding = padding, bias = bias) 
        self.def_2 = DefBlock(channels = hidden_channels, offset_channels = hidden_channels,
                                deform_groups = deform_groups, kernel_size = kernel_size,
                                padding = padding, bias = bias) 
        self.def_3 = DefBlock(channels = hidden_channels, offset_channels = hidden_channels,
                                deform_groups = deform_groups, kernel_size = kernel_size,
                                padding = padding, bias = bias) 
        self.def_4 = DefBlock(channels = hidden_channels, offset_channels = hidden_channels,
                                deform_groups = deform_groups, kernel_size = kernel_size,
                                padding = padding, bias = bias) 

        # Output layer
        self.reconstruct_layer = nn.Conv2d(hidden_channels, output_channels, 
                                            kernel_size = kernel_size, padding = padding, bias = bias)

    def align(self, x:torch.Tensor, x_ref:torch.Tensor) -> torch.Tensor:
        '''
            Use the deformable layers to align x to x_ref:
                x [B x F x H x W] input features to be deformed/aligned
                x_ref [B x F x H x W] features to be deformed/aligned too
            NOTE: F is hidden_channels from constructor
        '''
        x_feat = self.bottleneck(torch.cat([x, x_ref], dim=1))
        x_feat = self.def_1(x_feat, x_feat)
        x_feat = self.def_2(x_feat, x_feat)
        x = self.def_3(x, x_feat) # Applies deformation to x here
        return self.def_4(x, x) 

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
            Applies align net to the images in x
                x [B x num_imgs*C x H x W] input images that will be aligned 
                    where the images are stored along dim=1. So the i-th image is:
                        x[:, i*C:(i+1)*C, :, :]
                    The first image is the reference image being aligned to.
                    So out[:, 0:C, :, :] = x[:, 0:C, :, :]
        '''

        # Extract features
        num_imgs = int(x.shape[1]/self.input_channels)
        x_feat = x.reshape((-1, self.input_channels, x.shape[-2], x.shape[-1]))
        x_feat = self.conv_first(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.residual_layer(x_feat)
        x_feat = x_feat.reshape((-1, num_imgs*self.hidden_channels, x.shape[-2], x.shape[-1]))


        out:List[torch.Tensor] = []
        for img_idx in range(num_imgs):
            # First image is reference image so just pass to output
            if img_idx == 0:
                out.append(x[:, 0:self.input_channels, :, :])
            # Align and reconstruct features from other images
            else:
                x_feat_aligned = self.align(x_feat[:, img_idx*self.hidden_channels \
                                                        :(img_idx+1)*self.hidden_channels, :, :],
                                            x_feat[:, 0:self.hidden_channels, :, :])
                out.append(self.reconstruct_layer(x_feat_aligned))

        return torch.cat(out, dim=1)

    def getOutputSize(self, input_size:torch.Size) -> torch.Size:
        '''
            Returns the size of output given an input size
                Input: input_size of tuple size (B, C, H, W)
                Output: output_size of tuple size (B, C_out, H_out, W_out)
        '''
        return input_size


class AlignUNet(nn.Module):
    '''
        Network that combines TDAN alignment and U-Net
            input_channels: number of channels for input image
            output_channels: number of channels for output image
            num_images: number of images per forward pass
            layer_num_channels: number of channels in the internal layers (will be mirrored for encoder/decoder)
            drop_out_layers_enc: drop out probability for encoder layers. Note: zero implies no dropout
                            Should be the same length as layer_num_channels
            drop_out_layers_dec: drop out probability for decoder layers. Note: zero implies no dropout
                            Should be the same length as layer_num_channels
    '''
    def __init__(self, input_channels:int, 
                output_channels:int,
                num_images:int,
                layer_num_channels:List[int] = [64, 128, 256, 512, 512, 512],
                drop_out_layers_enc:List[float] = [0, 0, 0, 0, 0, 0],
                drop_out_layers_dec:List[float] = [0.5, 0.5, 0.5, 0, 0, 0],
                ):
        super().__init__()
        self.align_net = AlignNet(input_channels, output_channels, 
                                    hidden_channels = 64,
                                    num_feat_extraction_blocks = 5,
                                    bias = True)
        
        self.u_net = UNet(input_channels * num_images, output_channels,
                            layer_num_channels = layer_num_channels,
                            drop_out_layers_enc = drop_out_layers_enc,
                            drop_out_layers_dec = drop_out_layers_dec,
                            kernel_size = 4,
                            leaky_relu_encoder = True,
                            leaky_relu_decoder = False,
                            batch_norm = True)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            x [B x num_images * C_in x H x W] where C_in is input_channels
            out [B x C_out x H x W], [B x num_images * C_in x H x W] where C_out is output_channels
        '''
        x = self.align_net(x)
        return self.u_net(x), x
    
    def getOutputSize(self, input_size:torch.Size) -> Tuple[torch.Size, torch.Size]:
        '''
            Returns the size of output given an input size
                Input: input_size of tuple size (B, C, H, W)
                Output: output_size of tuple size (B, C_out, H_out, W_out)
        '''
        align_size = self.align_net.getOutputSize(input_size)
        return self.u_net.getOutputSize(align_size), align_size