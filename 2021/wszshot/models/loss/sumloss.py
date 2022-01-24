import torch
import torch.nn as nn

from utils.generic.getaccess import get_class_from_str
from copy import deepcopy
import pdb

class SumLoss(nn.Module):
    def __init__(self,
                 criterion_config):
        super(SumLoss, self).__init__()
        
        sum_loss = criterion_config['sum_loss']
        self.dict_losses   = criterion_config['losses']
        self.split_pred    = sum_loss['split_pred']
        self.split_target  = sum_loss['split_target']
        self.grad_stats    = sum_loss['grad_stats']
        
        if self.split_pred or self.split_target:
            assert all('pred_idx' in loss_config_dict.keys() for key, loss_config_dict in self.dict_losses.items()), ...
            "Error: Expecting indices corresponging to predictions in losses"
        if self.split_target:
            assert all('target_idx' in loss_config_dict.keys() for key, loss_config_dict in self.dict_losses.items()), ...
            "Error: Expecting an indices corresponging to targets in losses"
        
        # parse criterion_config to populate loss_config(s) with corresponding loss functions
        self.criterion_config = self.parse_criterion_config(criterion_config['losses'])
    
    def parse_criterion_config(self,
                               criterion_config):
        _criterion_config = {}
        for loss_key, loss_config in criterion_config.items():
            _loss_config = deepcopy(loss_config)
            weight       = _loss_config.pop('weight')
            pred_idx     = _loss_config.pop('pred_idx')
            target_idx   = _loss_config.pop('target_idx')
            class_name   = _loss_config.pop('function')
            loss_fn_inst = get_class_from_str(loss_config['function'])
            loss_func    = loss_fn_inst(_loss_config)
            
            _criterion_config[loss_key]               = {}
            _criterion_config[loss_key]['weight']     = weight
            _criterion_config[loss_key]['pred_idx']   = pred_idx
            _criterion_config[loss_key]['target_idx'] = target_idx
            _criterion_config[loss_key]['function']   = loss_func
        
        return _criterion_config
            
        
    def forward(self, pred, target):
        #pdb.set_trace()
        pred_mode = self.get_mode(pred, self.split_pred)
        if pred_mode == 'dict':
            for loss_key, loss_config_dict in self.dict_losses.items():
                if isinstance(loss_config_dict['pred_idx'], list): # loss requires multiple predictions
                    for loss_config_key in loss_config_dict['pred_idx']:
                        assert loss_config_key in pred.keys(), f"Error: Keys of pred do not match losses.keys {loss_config_key}"
                else: #apply loss requires single prediction
                    assert loss_config_dict['pred_idx'] in pred.keys(), ...
                    f"Error: Keys of pred do not match losses.keys() {loss_config_dict['pred_idx']}"
        elif pred_mode == 'list':
            for loss_key, loss_config_dict in self.dict_losses.items():
                if isinstance(loss_config_dict['pred_idx'], list): # loss requires multiple predictions
                    for loss_config_idx in loss_config_dict['pred_idx']:
                        assert loss_config_idx < len(pred), f"Error: Loss index {loss_config_idx} is out of range(len(prediction))"
                else: #apply loss requires single prediction
                    assert all(loss_config_dict['pred_idx'] < len(pred) for loss_key, loss_config_dict in self.dict_losses.items()), ...
                    f"Error: Loss index {loss_config_idx['pred_idx']} is out of range(len(prediction))"
        
        target_mode = self.get_mode(target, self.split_target)
        if target_mode == 'dict':
            for loss_key, loss_config_dict in self.dict_losses.items():
                if isinstance(loss_config_dict['target_idx'], list): # loss requires multiple targets
                    for loss_config_key in loss_config_dict['target_idx']:
                        assert loss_config_key in target.keys(), f"Error: Keys of target do not match losses.keys {loss_config_key}"
                else: ##apply loss requires single target
                    assert loss_config_dict['target_idx'] in target.keys(), ...
                    f"Error: Keys of target do not match losses.keys() {loss_config_dict['target_idx']}"
        elif target_mode == 'list':
            for loss_key, loss_config_dict in self.dict_losses.items():
                if isinstance(loss_config_dict['target_idx'], list): # loss requires multiple targets
                    for loss_config_idx in loss_config_dict['target_idx']:
                        assert loss_config_idx < len(target), f"Error: Loss index {loss_config_idx} is out of range(len(target))"
                else: #apply loss requires single target
                    assert all(loss_config_dict['target_idx'] < len(target) for loss_key, loss_config_dict in self.dict_losses.items()), ...
                    f"Error: Loss index {loss_config_idx['target_idx']} is out of range(len(target))"
        
        criterion = 0
        for loss_idx, (loss_key, loss_config) in enumerate(self.criterion_config.items()):
            weight = loss_config['weight']
            loss_fn = loss_config['function']
            _pred, _target = self.get_pred_target(pred_mode, target_mode, 
                                                  loss_config['pred_idx'], loss_config['target_idx'],
                                                  pred, target)
            #pdb.set_trace()
            criterion += weight * loss_fn(_pred, _target)
        return criterion
    
    def get_mode(self, value, split):
        '''
        Return the mode (dict/list/default) to handle the predictions and targets
        '''
        if isinstance(value, dict) and split:
            mode = 'dict'
        elif isinstance(value, list) and split:
            mode = 'list'
        else:
            mode = 'default'
        return mode
    
    def get_pred_target(self, 
                        pred_mode, target_mode, 
                        pred_idx, target_idx,
                        pred, target):
        '''
        Return the (prediction, target) as per the mode specified
        '''
        if pred_mode == 'dict' or pred_mode == 'list':
            if isinstance(pred_idx, list):
                _pred = []
                for key_idx in pred_idx:
                    _pred.append(pred[key_idx])
            else:
                _pred = pred[pred_idx]

        if target_mode == 'dict' or target_mode == 'list':
            if isinstance(target_idx, list):
                _target = []
                for key_idx in target_idx:
                    _target.append(target[key_idx])
            else:
                _target = target[target_idx]
        
        return _pred, _target