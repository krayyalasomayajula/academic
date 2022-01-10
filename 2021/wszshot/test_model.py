#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import yaml
import pdb


# # CLI config

# In[2]:


cli = {}
cli['config'] = './config/zs_config.yml'
cli['log_dir'] = './test_log'

if cli['log_dir'] is None:
        cli['log_dir'] = input ("Enter directory to save model and logs:")
    
if not os.path.exists(cli['log_dir']):
    os.makedirs(cli['log_dir'])
else:
    print(f"{cli['log_dir']} directory: exists")

with open(cli['config'], 'r') as f:
    args = yaml.safe_load(f)

print(args.keys())


# In[3]:


from utils import (_NATIVE_AMP_AVAILABLE, _TORCHVISION_AVAILABLE,
                  _GYM_AVAILABLE, _SKLEARN_AVAILABLE,
                  _PIL_AVAILABLE, _OPENCV_AVAILABLE,
                  _WANDB_AVAILABLE, _MATPLOTLIB_AVAILABLE,
                  _TORCHVISION_LESS_THAN_0_9_1, _PL_GREATER_EQUAL_1_4,
                  _PL_GREATER_EQUAL_1_4_5, _TORCH_ORT_AVAILABLE,
                  _TORCH_MAX_VERSION_SPARSEML, _SPARSEML_AVAILABLE)


# In[4]:


print(f"_NATIVE_AMP_AVAILABLE: {_NATIVE_AMP_AVAILABLE}")

print(f"_TORCHVISION_AVAILABLE: {_TORCHVISION_AVAILABLE}")
print(f"_GYM_AVAILABLE: {_GYM_AVAILABLE}")
print(f"_SKLEARN_AVAILABLE: {_SKLEARN_AVAILABLE}")
print(f"_PIL_AVAILABLE: {_PIL_AVAILABLE}")
print(f"_OPENCV_AVAILABLE: {_OPENCV_AVAILABLE}")
print(f"_WANDB_AVAILABLE: {_WANDB_AVAILABLE}")
print(f"_MATPLOTLIB_AVAILABLE: {_MATPLOTLIB_AVAILABLE}")
print(f"_TORCHVISION_LESS_THAN_0_9_1: {_TORCHVISION_LESS_THAN_0_9_1}")
print(f"_PL_GREATER_EQUAL_1_4: {_PL_GREATER_EQUAL_1_4}")
print(f"_PL_GREATER_EQUAL_1_4_5: {_PL_GREATER_EQUAL_1_4_5}")
print(f"_TORCH_ORT_AVAILABLE: {_TORCH_ORT_AVAILABLE}")
print(f"_TORCH_MAX_VERSION_SPARSEML: {_TORCH_MAX_VERSION_SPARSEML}")
print(f"_SPARSEML_AVAILABLE: {_SPARSEML_AVAILABLE}")


# In[5]:


import os
import numpy as np
from torch.utils.data import Dataset

#from pl_bolts.utils import _PIL_AVAILABLE
#from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    print(f"Warning missing pkg: Pillow")


# In[6]:


print(args['loaders'])
print(args['loaders'].keys())


# In[7]:


from utils.generic.parsing import recursive_parse_settings
from utils.generic.getaccess import get_class_from_str


# In[8]:


dataset_config = args['loaders']['dataset_config']
loader_config = args['loaders']['loader_config']
print(dataset_config)
print(loader_config)


# In[9]:


loader_key, loader_config_value = next(iter(loader_config.items()))
loader_class = get_class_from_str(loader_key)
phosc_loader = loader_class(dataset_config, loader_config_value)
phosc_loader.prepare_data()

trainset_sz = len(phosc_loader.trainset)
validset_sz = len(phosc_loader.validset)
print(f"size(trainset): {trainset_sz}, size(validset): {validset_sz}")


# # Test <code>dataset</code>

# In[10]:


import math
def get_data_from_dataset(dataset, n_epochs, steps_per_epoch, batch_size):
    for epoch_idx in range(n_epochs):
        for batch_idx in range(steps_per_epoch):
            batch_dict = {}
            for idx in range(batch_size):
                sample = next(iter(dataset))
                if len(batch_dict) ==0 :
                    for key, value in sample.items():
                        batch_dict[key] = [value]
                else:
                    for key, value in sample.items():
                        batch_dict[key] += [value]
            
            for key, value in batch_dict.items():
                batch_dict[key] = np.array(value)
                
            '''print(f"batch_idx: {batch_idx}, sample_batch: {batch_dict.keys()}")
            
            for key,value in batch_dict.items():
                print(f"key: {value.shape}")'''

get_data_from_dataset(phosc_loader.trainset, 
                      3, 
                      math.ceil(trainset_sz/loader_config_value['batch_size']), 
                      loader_config_value['batch_size'])
get_data_from_dataset(phosc_loader.validset, 
                      3, 
                      math.ceil(trainset_sz/loader_config_value['batch_size']), 
                      loader_config_value['batch_size'])


# # Test <code>loader</code>

# In[11]:


#Plotting a Batch of DataLoader
import torch
import matplotlib.pyplot as plt

SHOW_INPUT=False
if SHOW_INPUT:
    plt.figure(figsize = (8,16))
    fig, axes = plt.subplots(2, 5) # factors of batch_size
    for batch_idx, batch in enumerate(phosc_loader.train_dataloader()):
        for key, value in batch.items():
            print(f"idx: {batch_idx}, key: {key}, shape: {value.shape}")

        #for e,(img, lbl) in enumerate(zip(images, labels)):
        images = (torch.permute(batch['img'], (0, 2, 3, 1))*255.0).numpy().astype('uint8')
        labels = phosc_loader.wordLabelEncoder.inverse_transform(batch['wlabel'].numpy().tolist())
        print(labels)
        for idx in range(images.shape[0]):
            img = images[idx, :, :, :]

            plt.subplot(2, 5, idx+1)
            plt.imshow(img)
            plt.title(f'{labels[idx]}')

        plt.tight_layout()
        plt.show()
        val = 'y' #0
        while val != 'y':
            val = input("Enter y to continue: ")

        plt.clf()


# # Test <code>model</code>

# ## Print <code>model</code>

# In[12]:


def append_key_to_subconfig(main_key, subconfig):
    config = {}
    for key, value in subconfig.items():
        new_key = '_'.join([main_key, key])
        config[new_key] = value
    return config

_model_config = next(iter(args['model'].values()))
model_config = {}
    
for key, value in _model_config.items():
    _config = append_key_to_subconfig(key, value)
    model_config.update(_config)
print(model_config)


# In[13]:


from models.phocnet.phosc import PHOSCNet
conv_in_ch = model_config.pop('conv_in_ch')
conv_fmaps = model_config.pop('conv_fmaps')
model = PHOSCNet(conv_in_ch, conv_fmaps, **model_config)

PRINT_MODEL=False
if PRINT_MODEL:
    print(model.cnn_arch)


# ## Run <code>model</code>

# In[14]:


RUN_MODEL=False
if RUN_MODEL:
    for batch_idx, batch in enumerate(phosc_loader.train_dataloader()):
        cnn_feats = model.forward(batch['img'])
        print(cnn_feats['phos'].shape, cnn_feats['phoc'].shape)


# In[15]:


criterion_config = args['criterion']
print(criterion_config)


# # Test <code>loss</code>

# In[16]:


from models.loss.sumloss import SumLoss
assert 'sum_loss' in criterion_config.keys(), "Error: Include sum_loss config in config (required of single loss case)"
sum_loss_cls = SumLoss(criterion_config)


# ## Run <code>loss</code>

# In[17]:


RUN_MODEL_WITH_LOSS=False
if RUN_MODEL_WITH_LOSS:
    for batch_idx, batch in enumerate(phosc_loader.train_dataloader()):
            #xx = torch.permute(batch['img'], (0, 3, 1, 2)).float()/255.0
            cnn_feats = model.forward(batch['img'])
            #print(cnn_feats['phos'].shape, cnn_feats['phoc'].shape)
            loss = sum_loss_cls(cnn_feats, batch)
            print(f"batch[{batch_idx}], loss: {loss}")


# # Test <code>metric</code>

# In[18]:


#from torchmetrics import CosineSimilarity
#from models.metric.wsmetric import WSMetric


# # Run <code>metric</code>

# In[19]:


from termcolor import colored

def log_plots(img_batch, pred_word, target_word):
    N_COLS = 5
    N_ROWS = math.ceil(img_batch.shape[0]/N_COLS)
    SCALE = 2
    FIG_HT, FIG_WD = 4, 6
    plt.figure()
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize = (SCALE*FIG_WD, SCALE*FIG_HT)) # factors of batch_size
    for idx in range(img_batch.shape[0]):
        img = img_batch[idx, :, :, :]

        _fig = plt.subplot(N_ROWS, N_COLS, idx+1)
        plt.imshow(img)
        plt.title(f'({pred_word[idx]}, {target_word[idx]})', fontsize=15)
        #_fig.text(5, 1, f'{pred_word[idx]}', ha="center", va="bottom", fontsize=15, color="blue")
        #_fig.text(15, 1, f'{target_word[idx]}', ha="center", va="bottom", fontsize=15, color="green")
        
        plt.axis('off')

    plt.tight_layout(pad=3.0)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.cla()
    plt.close()
    return data
    


# In[20]:


from copy import deepcopy
from utils.generic.getaccess import get_class_from_str

class_fn = get_class_from_str(args['metric']['name'])
ws_metric = class_fn(phosc_loader.df_all, phosc_loader.wordLabelEncoder)

RUN_METRIC_ON_BATCH = False
if RUN_METRIC_ON_BATCH:
    for batch_idx, batch in enumerate(phosc_loader.train_dataloader()):
        pred   = deepcopy(batch)
        target = deepcopy(batch)
        img_batch = target.pop('img')
        img_batch = (torch.permute(img_batch, (0, 2, 3, 1))*255.0).numpy().astype('uint8')
        _metric = ws_metric.compute(pred, target)
        
        '''disp_img = log_image_tiles(img_batch, _metric['accuracy_word'].tolist(), 
                        phosc_loader.wordLabelEncoder.inverse_transform(target['wlabel'].tolist()))'''
        disp_img = log_plots(img_batch, _metric['accuracy_word'].tolist(), 
                        phosc_loader.wordLabelEncoder.inverse_transform(target['wlabel'].tolist()))
        im = Image.fromarray(disp_img)
        im.save(f"your_file_{batch_idx}.png")


# # Setup <code>LightningModule</code>

# In[21]:


from pytorch_lightning import LightningModule, Trainer, seed_everything

class PhoscWSTask(LightningModule):
    def __init__(self,
                 model,
                 loss,
                 metric,
                 trainer_config):
        super().__init__()
        #pdb.set_trace()
        self.model  = model
        self.loss   = loss
        self.metric = metric
        
        self.num_tile_col =  trainer_config['num_tile_col']
        self.log_train_every = trainer_config['intervals']['log_train_every']
        self.validate_every = trainer_config['intervals']['validate_every']
        self.optimizer = trainer_config['optimizer']
        assert 'name' in self.optimizer.keys(), 'Error: Optimizer needs to be selected'
        assert self.optimizer['name'] == 'Adam', 'Error: Only Adam optimizer implemeted in trainer'
        
    def training_step(self, batch, batch_idx):
        #pdb.set_trace()
        logging = self.global_step % self.log_train_every == 0
        dict_values = self._shared_step(batch, batch_idx, logging)
        ret_values  = self._populate_return_values(dict_values)
        self._populate_tensorboard_logs(ret_values, stage='train', logging=logging)
        
        step_dict={
            # required
            'loss': ret_values['loss']}
        print("At training_step, global_step=", self.global_step)
        return step_dict 
        
        
    def validation_step(self, batch, batch_idx):
        #pdb.set_trace()
        dict_values = self._shared_step(batch, batch_idx, True)
        ret_values  = self._populate_return_values(dict_values)
        self._populate_tensorboard_logs(ret_values, stage='validation', logging=True)
        
        step_dict={
            # required
            'loss': ret_values['loss']}
    
        print("At validation_step, global_step=", self.global_step)
        return step_dict 
        
        
    def _shared_step(self, batch, batch_idx, logging):
        img = batch.pop('img')
        cnn_feats = self.model(img)
        loss = self.loss(cnn_feats, batch)
        acc = self.metric.compute(cnn_feats, batch)
        if logging:
            img_batch = (torch.permute(img, (0, 2, 3, 1))*255.0).cpu().numpy().astype('uint8')
            tiled_img = self._log_image_tiles(img_batch, acc['accuracy_word'].tolist(), 
                        self.metric.wordLabelEncoder.inverse_transform(batch['wlabel'].tolist()))
        else:
            tiled_img = None
        
        return {'loss': loss, 
                'accuracy': acc, 
                'prediction': tiled_img}

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.optimizer['lr'], betas=self.optimizer['betas'])
    
    
    def test_step(self, batch, batch_idx):
        dict_values = self._shared_step(batch, batch_idx, True)
        ret_values  = self._populate_return_values(dict_values)
        self._populate_tensorboard_logs(ret_values, stage='test')
        
        step_dict={
            # required
            'loss': ret_values['loss']}
    
        print("At test_step, global_step=", self.global_step)
        return step_dict 
        

    '''
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat
    '''
    
    
    def _log_image_tiles(self, img_batch, pred_word, target_word):
        N_COLS = self.num_tile_col
        N_ROWS = math.ceil(img_batch.shape[0]/N_COLS)
        SCALE = 2
        FIG_HT, FIG_WD = 4, 6

        plt.figure()
        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize = (SCALE*FIG_WD, SCALE*FIG_HT)) # factors of batch_size
        for idx in range(img_batch.shape[0]):
            img = img_batch[idx, :, :, :]

            _fig = plt.subplot(N_ROWS, N_COLS, idx+1)
            plt.imshow(img)
            plt.title(f'({pred_word[idx]}, {target_word[idx]})', fontsize=15)
            plt.axis('off')

        plt.tight_layout(pad=3.0)
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        np_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        np_arr = np_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.cla()
        plt.close('all')
        return np_arr

    def _populate_return_values(self,
                               dict_values):
        ret_values = {}
        for key, value in dict_values.items():
            if key == 'accuracy': #accuracy returns a dict of values
                for _key, _value in value.items():
                    if _key != 'accuracy_word':
                        ret_values[_key] = _value
            else:
                ret_values[key] = value
        
        return ret_values
        
    def _populate_tensorboard_logs(self,
                                   ret_values,
                                   stage='train',
                                   logging=False):
        ''' ret_values keys : 
        scalars: 'loss', 'similarity_phos', 'similarity_phoc', 'similarity_phosc', 'accuracy_phos', 'accuracy_phoc', 'accuracy_phosc'
        image: 'prediction'
        '''
        # logging using tensorboard logger
        if logging:
            for key, value in ret_values.items():
                if key == 'prediction':
                    self.logger.experiment.add_image(f'{stage}_{key}', value, self.global_step, dataformats="HWC")
                else:
                    self.logger.experiment.add_scalar(f'{stage}_{key}', value, self.global_step)
        


# In[ ]:


from pytorch_lightning.loggers import TensorBoardLogger


print(torch.cuda.is_available())
if 1:
    seed_everything(42)
    trainer_config = args['trainer']
    pl_phosc_model = PhoscWSTask(model,
                 sum_loss_cls,
                 ws_metric,
                 trainer_config)
    phosc_logger = TensorBoardLogger("tb_logs", name="ws_phosc_default")
    trainer = Trainer(gpus=1,
                      limit_val_batches=1,
                      val_check_interval=trainer_config['intervals']['validate_every'],
                      progress_bar_refresh_rate=0,
                      max_epochs=trainer_config['max_epochs'],
                      logger=phosc_logger)
    trainer.fit(pl_phosc_model, phosc_loader.train_dataloader(), phosc_loader.val_dataloader())
    

