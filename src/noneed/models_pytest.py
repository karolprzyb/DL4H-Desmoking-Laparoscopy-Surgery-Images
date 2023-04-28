import torch
from models import *

def test_conv_relu_no_crop_output():
    in_channels = 3
    conv_relu = ConvRelu(in_channels=in_channels, out_channels=64, 
                            kernel_size = 4,
                            num_conv = 2,
                            drop_out = 0.3,
                            dilation = 2,
                            stride = 2,
                            padding = 2,
                            crop_output = False)
    input = torch.randn(32, in_channels, 50, 100)

    assert conv_relu(input).shape == conv_relu.getOutputSize(input.shape)

def test_conv_relu_crop_output():
    in_channels = 256
    kernel_size = 3
    conv_relu = ConvRelu(in_channels=in_channels, out_channels=512, 
                            kernel_size = kernel_size,
                            num_conv = 3,
                            drop_out = 0.0,
                            dilation = 1,
                            stride = 1,
                            padding = int(np.ceil((kernel_size - 1)/2.0)),
                            crop_output = True)
    input = torch.randn(16, in_channels, 256, 256)

    assert conv_relu(input).shape == conv_relu.getOutputSize(input.shape)

def test_res_block():
    channels  = 64
    res_block = ResBlock(channels = channels)
    input = torch.randn(32, channels, 128, 128)

    assert res_block(input).shape == res_block.getOutputSize(input.shape)

def test_def_block():
    channels  = 64
    offset_channels = 256
    def_block = DefBlock(channels = channels,
                        offset_channels = offset_channels,
                        deform_groups = 8,
                        kernel_size = 3,
                        padding = 1,
                        bias = True)
    input = torch.randn(32, channels, 128, 128)
    offset_input = torch.randn(32, offset_channels, 128, 128)

    assert def_block(input, offset_input).shape == def_block.getOutputSize(input.shape)

def test_down():
    in_channels = 128
    down = Down(in_channels=in_channels, out_channels=256, 
                kernel_size = 6,
                num_conv = 1,
                drop_out = 0.0,
                leaky_relu = True,
                batch_norm = True,
                padding = 2)
    input  = torch.randn(1, in_channels, 128, 128)
    output = down(input)
    assert output.shape == torch.Size([1, 256, 64, 64])
    assert output.shape == down.getOutputSize(input.shape)

def test_up():
    in_channels = 128
    up = Up(in_channels=in_channels, out_channels=64, 
                kernel_size = 4,
                num_conv = 1,
                drop_out = 0.5,
                leaky_relu = False,
                batch_norm = True)
    input  = torch.randn(4, in_channels, 64, 64)
    output = up(input)
    assert output.shape == torch.Size([4, 64, 128, 128])
    assert output.shape == up.getOutputSize(input.shape)


def test_encoder():
    input_channels = 3
    layers = [64, 128, 256, 512, 512, 512]
    output_channels = 512
    res = 512
    batch = 4
    enc = Encoder(input_channels = input_channels,
                  output_channels = output_channels,
                  layer_num_channels = layers,
                  drop_out_layers = [0 for _ in range(len(layers))],
                  save_skips = True,
                  kernel_size = 4,
                  leaky_relu = True,
                  batch_norm = True,
                  use_max_pool=False)
    input  = torch.randn(batch, input_channels, res, res)
    output = enc(input)

    output_shapes = [torch.Size([batch, layers[layer_idx], int(res/2**layer_idx), int(res/2**layer_idx)]) \
                        for layer_idx in range(len(layers))]

    assert output.shape == torch.Size([batch, output_channels, 
                                        int(res/2**len(layers)), int(res/2**len(layers))])
    assert output.shape == enc.getOutputSize(input.shape)
    for layer_idx in range(len(layers)):
        assert getattr(enc, "skip_connection_{}".format(layer_idx)).shape \
                    == output_shapes[layer_idx]
        assert getattr(enc, "skip_connection_{}".format(layer_idx)).shape \
                    == enc.getOutputSizeSkip(input.shape, layer_idx)

def test_decoder_no_skip():
    input_channels = 512
    layers = [512, 512, 512, 256, 128, 64]
    dec = Decoder(input_channels = input_channels,
                    output_channels = 3, 
                    layer_num_channels = layers,
                    drop_out_layers = [0.5, 0, 0, 0, 0, 0],
                    skip_layer_inputs = False,
                    skip_layer_mult = 2,
                    kernel_size = 4,
                    leaky_relu = False,
                    batch_norm = True)
    input  = torch.randn(8, input_channels, 8, 8)
    output = dec(input)
    assert output.shape == torch.Size([8, 3, 8*2**len(layers), 8*2**len(layers)])
    assert output.shape == dec.getOutputSize(input.shape)
    
def test_unet():
    input_channels = 3
    net = UNet(input_channels=input_channels, 
                output_channels=1, 
                layer_num_channels = [64, 128, 256, 512, 512, 512],
                drop_out_layers_enc = [0, 0, 0, 0, 0, 0],
                drop_out_layers_dec = [0.5, 0.5, 0.5, 0, 0, 0],
                kernel_size = 4,
                leaky_relu_encoder = True,
                leaky_relu_decoder = False,
                batch_norm = True)
    input  = torch.randn(1, input_channels, 256, 256)
    output = net(input)
    assert output.shape == net.getOutputSize(input.shape)

def test_unet_2():
    input_channels = 1
    net = UNet(input_channels=input_channels, 
                output_channels=3, 
                layer_num_channels = [64, 128, 256],
                drop_out_layers_enc = [0, 0, 0],
                drop_out_layers_dec = [0, 0, 0],
                kernel_size = 4,
                leaky_relu_encoder = False,
                leaky_relu_decoder = True,
                batch_norm = True)
    input  = torch.randn(1, input_channels, 256, 256)
    output = net(input)
    assert output.shape == net.getOutputSize(input.shape)

def test_align_net():
    channels = 3
    num_img = 4
    net = AlignNet(channels, channels,
                    hidden_channels = 64,
                    num_feat_extraction_blocks = 5,
                    bias = True)
    input  = torch.randn(2, channels*num_img, 256, 256)
    output = net(input)
    assert output.shape == net.getOutputSize(input.shape)

def test_align_unet():
    channels = 2
    num_img = 5
    net = AlignUNet(channels, channels, num_img)
    input  = torch.randn(4, channels*num_img, 128, 128)
    output, align = net(input)
    assert output.shape == net.getOutputSize(input.shape)[0]
    assert align.shape == net.getOutputSize(input.shape)[1]
