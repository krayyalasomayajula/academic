from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F

'''
Based on implementations from
https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/buildingblocks.py
'''

def conv2d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, conv_type, kernel_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            if conv_type == '2D':
                modules.append(('conv', conv2d(in_channels, out_channels, kernel_size, bias, padding=padding)))
            elif conv_type == '3D':
                modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
            else:
                raise ValueError(f"Unsupported conv type '{conv_type}'. MUST be one of ['2D', '3D']")
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, conv_type, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, conv_type, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, conv_type, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, conv_type, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, conv_type, kernel_size, order, num_groups,
                                   padding=padding))


def create_phocnet_architecture(conv_type, 
                                in_channels, 
                                conv_fmaps, 
                                conv_kernel_size, 
                                conv_padding, 
                                conv_layer_order, 
                                conv_num_groups,
                                pool_indices,
                                pool_kernel_size,
                                pool_stride,
                                pool_padding,
                                last_single_conv=None):
    
    # create CNN layers
    cnn_layers = []
    for i, out_feature_num in enumerate(conv_fmaps):
        if i == 0:
            layer = DoubleConv(in_channels, 
                               out_feature_num, 
                               False,
                               conv_type, 
                               kernel_size=conv_kernel_size, 
                               order=conv_layer_order, 
                               num_groups=conv_num_groups, 
                               padding=conv_padding)
        else:
            layer = DoubleConv(conv_fmaps[i - 1], 
                               out_feature_num,
                               False, 
                               conv_type, 
                               kernel_size=conv_kernel_size, 
                               order=conv_layer_order, 
                               num_groups=conv_num_groups, 
                               padding=conv_padding)
        
        cnn_layers.append((f'DoubleConv_{i+1}', layer))
    
    if last_single_conv is not None:
        layer = SingleConv(conv_fmaps[-1], 
                           last_single_conv, 
                           conv_type, 
                           kernel_size=conv_kernel_size, 
                           order=conv_layer_order, 
                           num_groups=conv_num_groups, 
                           padding=conv_padding)
        cnn_layers.append(('SingleConv_final', layer))
    
    # create pooling layers
    pooling_layers = []
    for i, _ in enumerate(pool_indices):
        layer = nn.MaxPool2d(pool_kernel_size, stride=pool_stride, padding=pool_padding)
        pooling_layers.append((f'MaxPooling_{i+1}', layer))
    
    # create CNN feature backbone. Depth of the layers is equal to `len(conv_fmaps) + len(max_pooling_indices)`
    cnn_arch = nn.ModuleList()
    cnn_layer_idx = 0
    for arch_layer_idx in range(len(cnn_layers) + len(pooling_layers)):
        if arch_layer_idx in pool_indices:
            name, pool_layer = pooling_layers[pool_indices.index(arch_layer_idx)]
            cnn_arch.add_module(name, pool_layer)
        else:
            name, layer = cnn_layers[cnn_layer_idx]
            cnn_arch.add_module(name, layer)
            cnn_layer_idx += 1
    
    return cnn_arch
