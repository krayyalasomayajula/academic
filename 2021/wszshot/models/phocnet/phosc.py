import torch.nn as nn
from models.phocnet.buildingblocks import create_phocnet_architecture
from models.phocnet.pyramidpooling import SpatialPyramidPooling, TemporalPyramidPooling
from utils import _PL_GREATER_EQUAL_1_4_5

import pdb

class PhoscFeatureHead(nn.Module):
    def __init__(self, fmaps, dropouts, ftype):
        super(PhoscFeatureHead, self).__init__()
        assert len(fmaps) >= len(dropouts)
        self.fmaps = fmaps
        self.dropouts = dropouts
        self.ftype = ftype

    def forward(self, x):
        return self.feature_head(x, self.fmaps, self.dropouts)
    
    #@staticmethod
    def feature_head(self, 
                     previous_conv, 
                     list_fmaps, list_dropouts):
        previous_conv_size = previous_conv.size(-1) # Expecting a flattened feature vector (N * dim)
        for i in range(len(list_fmaps)):
            if i < len(list_dropouts):
                if i == 0:
                    x= nn.Linear(previous_conv_size, list_fmaps[i]).cuda()(previous_conv)
                else:
                    x= nn.Linear(list_fmaps[i-1], list_fmaps[i]).cuda()(x)
                x = nn.ReLU().cuda()(x)
                x = nn.Dropout(list_dropouts[i]).cuda()(x)
            else: #No more dropuouts
                x= nn.Linear(list_fmaps[i-1], list_fmaps[i]).cuda()(x)
                if self.ftype == 'phos':
                    x = nn.ReLU().cuda()(x)
        return x

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
        self.phos_head = PhoscFeatureHead(phos_head_fmaps, phos_head_dropout, 'phos')
        self.phoc_head = PhoscFeatureHead(phoc_head_fmaps, phoc_head_dropout, 'phoc')
        

    def forward(self, x):
        for idx, layer in enumerate(self.cnn_arch):
            x = layer(x)
            #print(f'Layer{idx}: {x.shape}')
        x = self.pp_layer.forward(x)
        #print(f'PP layer: {x.shape}')
        phos_feat = self.phos_head(x)
        phoc_feat = self.phoc_head(x)
        ret_dict = {'phos': phos_feat, 
                'phoc': phoc_feat}
        return ret_dict

if _PL_GREATER_EQUAL_1_4_5:
    import pytorch_lightning as pl
    
    class plPHOSCNet(pl.LightningModule):
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
            super(plPHOSCNet, self).__init__()

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
            self.phos_head = PhoscFeatureHead(phos_head_fmaps, phos_head_dropout)
            self.phoc_head = PhoscFeatureHead(phoc_head_fmaps, phoc_head_dropout)


        def forward(self, x):
            for idx, layer in enumerate(self.cnn_arch):
                x = layer(x)
                #print(f'Layer{idx}: {x.shape}')
            x = self.pp_layer.forward(x)
            phos_feat = self.phos_head(x)
            phoc_feat = self.phoc_head(x)
            ret_dict = {'phos': phos_feat, 
                    'phoc': phoc_feat}
            return ret_dict

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)
