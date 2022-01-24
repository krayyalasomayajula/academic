import os
import yaml
import pdb
import math
import torch
import matplotlib.pyplot as plt
from utils import (_NATIVE_AMP_AVAILABLE, _TORCHVISION_AVAILABLE,
                  _GYM_AVAILABLE, _SKLEARN_AVAILABLE,
                  _PIL_AVAILABLE, _OPENCV_AVAILABLE,
                  _WANDB_AVAILABLE, _MATPLOTLIB_AVAILABLE,
                  _TORCHVISION_LESS_THAN_0_9_1, _PL_GREATER_EQUAL_1_4,
                  _PL_GREATER_EQUAL_1_4_5, _TORCH_ORT_AVAILABLE,
                  _TORCH_MAX_VERSION_SPARSEML, _SPARSEML_AVAILABLE)


import os
import numpy as np
from termcolor import colored


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

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    print(f"Warning missing pkg: Pillow")


#==============================================================================================
# # =============================== <code>lightning class</code> ==============================
#==============================================================================================


from pytorch_lightning import LightningModule, Trainer, seed_everything
import statistics

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
        self.log_valid_every = trainer_config['intervals']['log_valid_every']
        self.validate_every = trainer_config['intervals']['validate_every']
        self.validate_for = trainer_config['intervals']['validate_for']
        self.optimizer = trainer_config['optimizer']
        assert 'name' in self.optimizer.keys(), 'Error: Optimizer needs to be selected'
        assert self.optimizer['name'] == 'Adam', 'Error: Only Adam optimizer implemeted in trainer'
        
        print('================================== Model parameters ==================================')        
        for name, param in model.named_parameters():
            print(name, param.size())

        
        self.prev_avg_val_loss = np.inf
        self.val_loss   = []
        self.val_metric = []
        
        
        
    def training_step(self, batch, batch_idx):
        #pdb.set_trace()
        logging = self.global_step % self.log_train_every == 0 and self.global_step != 0
        dict_values = self._shared_step(batch, batch_idx, logging)
        ret_values  = self._populate_return_values(dict_values)
        self._populate_tensorboard_logs(ret_values, stage='train', logging=logging)
        
        step_dict={
            # required
            'loss': ret_values['loss']}
        #print("At training_step, global_step=", self.global_step)
        print(f"At train global_step={self.global_step}, loss={ret_values['loss'].item()}")
        return step_dict 
        
        
    def validation_step(self, batch, batch_idx):
        #pdb.set_trace()
        logging = batch_idx == self.validate_for - 1 # Log one batch in validation
        dict_values = self._shared_step(batch, batch_idx, logging)
        ret_values  = self._populate_return_values(dict_values)
        self._populate_tensorboard_logs(ret_values, stage='validation', logging=logging)
        
        #pdb.set_trace()
        self.val_loss.append(ret_values['loss'])
        self.val_metric.append(ret_values['accuracy_phosc'])
        
        step_dict={
            # required
            'loss': ret_values['loss'].item()}
    
        #print("At validation_step, global_step=", self.global_step)
        print(f"At validation batch_idx={batch_idx}, loss={ret_values['loss'].item()}")
        if batch_idx == self.validate_for - 1:
            #pdb.set_trace()
            avg_val_loss = torch.mean(torch.stack(self.val_loss))
            avg_val_accuracy = statistics.mean(self.val_metric)
            
            self.log('avg_val_loss', avg_val_loss)
            self.logger.experiment.add_scalar(f'avg_val_loss', avg_val_loss, self.global_step)
            self.logger.experiment.add_scalar(f'avg_val_accuracy', avg_val_accuracy, self.global_step)
            
            self.val_loss = []
            self.val_metric = []
            
            print(colored(f"avg. validation loss = {avg_val_loss.item()}, avg. validation accuracy = {avg_val_accuracy}", 'green'))
            if self.prev_avg_val_loss > avg_val_loss.item():
                print(colored(f"Saving model, validation loss = {avg_val_loss.item()} < {self.prev_avg_val_loss}", 'yellow'))
                self.prev_avg_val_loss = avg_val_loss.item()
            else:
                print(colored(f"Validation loss = {avg_val_loss.item()} >= {self.prev_avg_val_loss}", 'yellow'))
    
        return step_dict 
        
    #def on_validation_batch_end(self, batch, batch_idx):
    #    pdb.set_trace()
    #    if self.global_step % self.log_valid_every == 0 and self.global_step != 0:
    #        avg_val_loss = statistics.mean(self.val_loss)
    #        avg_val_accuracy = statistics.mean(self.val_metric)

    #        self.logger.experiment.add_scalar(f'avg_val_loss', avg_val_loss, self.global_step)
    #        self.logger.experiment.add_scalar(f'avg_val_accuracy', avg_val_accuracy, self.global_step)
    #        
    #        self.val_loss = []
    #        self.val_metric = []
    
    #def on_after_backward(self):
    #    # example to inspect gradient information in tensorboard
    #    if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
    #        params = self.state_dict()
    #        for name, grads in params.items():
    #            self.logger.experiment.add_histogram(tag=name, values=grads,
    #                                                 global_step=self.trainer.global_step)

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
        FIG_HT, FIG_WD = SCALE*4, SCALE*6

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
                    if isinstance(ret_values['similarity_phos'], torch.Tensor):
                        self.logger.experiment.add_scalar(f'{stage}_{key}', value.item(), self.global_step)
                    else:
                        self.logger.experiment.add_scalar(f'{stage}_{key}', value, self.global_step)
if __name__ == '__main__':
    print(torch.cuda.is_available())
    # # ================================== PL imports ================================
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    
    #==============================================================================================
    # # ================================== <code>CLI config</code> ================================
    #==============================================================================================

    cli = {}
    cli['config'] = './config/zs_config.yml'
    cli['log_dir'] = './ws_pl_logging'

    if cli['log_dir'] is None:
            cli['log_dir'] = input ("Enter directory to save model and logs:")

    if not os.path.exists(cli['log_dir']):
        os.makedirs(cli['log_dir'])
    else:
        print(f"{cli['log_dir']} directory: exists")

    with open(cli['config'], 'r') as f:
        args = yaml.safe_load(f)

    print(args.keys())

    print(args['loaders'])
    print(args['loaders'].keys())

    from utils.generic.parsing import recursive_parse_settings
    from utils.generic.getaccess import get_class_from_str


    dataset_config = args['loaders']['dataset_config']
    loader_config = args['loaders']['loader_config']
    print(dataset_config)
    print(loader_config)


    #==============================================================================================
    # # ================================== <code>DataLoader</code> ================================
    #==============================================================================================


    loader_key, loader_config_value = next(iter(loader_config.items()))
    loader_class = get_class_from_str(loader_key)
    phosc_loader = loader_class(dataset_config, loader_config_value)
    phosc_loader.prepare_data()

    trainset_sz = len(phosc_loader.trainset)
    validset_sz = len(phosc_loader.validset)
    print(f"size(trainset): {trainset_sz}, size(validset): {validset_sz}")


    #==============================================================================================
    # # ==================================== <code>model</code> ===================================
    #==============================================================================================

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


    from models.phocnet.phosc import PHOSCNet
    conv_in_ch = model_config.pop('conv_in_ch')
    conv_fmaps = model_config.pop('conv_fmaps')
    model = PHOSCNet(conv_in_ch, conv_fmaps, **model_config)

    PRINT_MODEL=True
    if PRINT_MODEL:
        print(model.cnn_arch)

    #==============================================================================================
    # # ==================================== <code>loss</code> ====================================
    #==============================================================================================

    criterion_config = args['criterion']
    print(criterion_config)

    from models.loss.sumloss import SumLoss
    assert 'sum_loss' in criterion_config.keys(), "Error: Include sum_loss config in config (required of single loss case)"
    sum_loss_cls = SumLoss(criterion_config)


    #==============================================================================================
    # # =================================== <code>metric</code> ===================================
    #==============================================================================================

    class_fn = get_class_from_str(args['metric']['name'])
    ws_metric = class_fn(phosc_loader.df_all, phosc_loader.wordLabelEncoder)


    #---------------------------------- lightning module class ----------------------------------
    seed_everything(42)
    trainer_config = args['trainer']
    pl_phosc_model = PhoscWSTask(model,
                 sum_loss_cls,
                 ws_metric,
                 trainer_config)
    #---------------------------------- TensorBoardLogger ----------------------------------
    phosc_logger = TensorBoardLogger(cli['log_dir'], name="ws_phosc_default")
    
    #---------------------------------- callbacks ----------------------------------
    checkpoint_callback = ModelCheckpoint(
    monitor="avg_val_loss",
    dirpath=cli['log_dir'],
    filename="ws_best_model-{epoch:03d}-{val_loss:.5f}",
    save_top_k=1,
    mode="min")
    
    trainer = Trainer(gpus=1,
                      limit_val_batches=pl_phosc_model.validate_for,
                      val_check_interval=pl_phosc_model.validate_every,
                      progress_bar_refresh_rate=0,
                      max_epochs=trainer_config['max_epochs'],
                      logger=phosc_logger,
                      callbacks=[checkpoint_callback])
    trainer.fit(pl_phosc_model, phosc_loader.train_dataloader(), phosc_loader.val_dataloader())
    