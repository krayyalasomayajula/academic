import torch.nn as nn
from models.phocnet.buildingblocks import create_phocnet_architecture
from models.phocnet.pyramidpooling import SpatialPyramidPooling, TemporalPyramidPooling

import pdb

def feature_head(list_fmaps, list_dropouts, ftype):
    #pdb.set_trace()
    layers = nn.ModuleList()
    linear_idx, relu_idx, dout_idx = 0, 0, 0
    for i in range(len(list_fmaps)):
        if i < len(list_dropouts):
            if i == 0:
                layers.add_module(f'fhead_linear{linear_idx}', nn.LazyLinear(list_fmaps[i])) # Expecting a flattened feature vector (N * dim)
            else:
                layers.add_module(f'fhead_linear{linear_idx}', nn.Linear(list_fmaps[i-1], list_fmaps[i]))
            linear_idx += 1

            layers.add_module(f'fhead_relu{relu_idx}', nn.ReLU())
            relu_idx += 1
            layers.add_module(f'fhead_drop{dout_idx}',nn.Dropout(list_dropouts[i]))
            dout_idx += 1
        else: #No more dropuouts
            layers.add_module(f'fhead_linear{linear_idx}', nn.Linear(list_fmaps[i-1], list_fmaps[i]))
            linear_idx += 1
            if ftype == 'phos':
                layers.add_module(f'fhead_relu{relu_idx}', nn.ReLU())
                relu_idx += 1
    return layers


class PHOSCNet(nn.Module):
    def __init__(self,
                 conv_in_ch,
                 conv_fmaps,
                 conv_type='2D',
                 conv_layer_order='gcr',
                 conv_num_groups=8, 
                 conv_kernel_size=3,
                 conv_stride=1,
                 conv_padding=1,
                 conv_last_single_conv_fmap=512,
                 
                 pool_layer_index=[1, 3],
                 pool_kernel_size=2,
                 pool_stride=2,
                 pool_padding=0,
                 
                 pyramid_pool_type='spatial',
                 pyramid_pool_levels=[1, 2, 4],
                 
                 phos_head_fmaps=[4096, 4096, 165],
                 phos_head_dropout=[0.5, 0.5],
                 
                 phoc_head_fmaps=[4096, 4096, 604],
                 phoc_head_dropout=[0.5, 0.5],
                 
                 **kwargs):
        super(PHOSCNet, self).__init__()

        assert isinstance(conv_fmaps, list) or isinstance(conv_fmaps, tuple)
        assert len(conv_fmaps) > 1, "Required at least 2 levels in the CNN feature extractor"
        
        # create CNN feature extractor architecture
        self.cnn_arch = create_phocnet_architecture(conv_type,
                                                    conv_in_ch,
                                                    conv_fmaps,
                                                    conv_kernel_size,
                                                    conv_padding,
                                                    conv_layer_order,
                                                    conv_num_groups,
                                                    pool_layer_index,
                                                    pool_kernel_size,
                                                    pool_stride,
                                                    pool_padding,
                                                    conv_last_single_conv_fmap)
        
        # create pyramid pooling layer
        if pyramid_pool_type == 'spatial':
            self.pp_layer = SpatialPyramidPooling(pyramid_pool_levels)
        elif pyramid_pool_type == 'temporal':
            self.pp_layer = TemporalPyramidPooling(pyramid_pool_levels)
        else:
            raise ValueError(f"Unsupported pyramid pooling type '{pyramid_pool_type}'. MUST be one of ['spatial', 'temporal']")
        
        # create phos, phoc heads
        self.phos_head = feature_head(phos_head_fmaps, phos_head_dropout, 'phos')
        self.phoc_head = feature_head(phoc_head_fmaps, phoc_head_dropout, 'phoc')
        

    def forward(self, x):
        for idx, layer in enumerate(self.cnn_arch):
            x = layer(x)
            #print(f'Layer{idx}: {x.shape}')
        x = self.pp_layer.forward(x)
        #print(f'PP layer: {x.shape}')
        for idx, layer in enumerate(self.phos_head):
            if idx == 0:
                phos_feat = layer(x)
            else:
                phos_feat = layer(phos_feat)
        
        for idx, layer in enumerate(self.phoc_head):
            if idx == 0:
                phoc_feat = layer(x)
            else:
                phoc_feat = layer(phoc_feat)
        
        ret_dict = {'phos': phos_feat, 
                'phoc': phoc_feat}
        return ret_dict
