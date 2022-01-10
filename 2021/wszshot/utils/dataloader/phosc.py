import os
from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

import pdb
import pandas as pd
import numpy as np
from copy import deepcopy
from PIL import Image
from sklearn import preprocessing

from utils.generic.getaccess import (get_class_from_key, 
                                     get_class_from_str, 
                                     get_filenames)
from utils.generic.parsing import recursive_parse_settings
from utils.phosc.phos_label_generator import gen_phos_label
from utils.phosc.phoc_label_generator import gen_phoc_label


class PHOSCZSDataModule(LightningDataModule):

    name = "phosczs"

    def __init__(
        self,
        _dataset_configs: dict = None,
        loader_config: dict = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """PHOSC zero-shot train, validation and test dataloaders.

        Note:
            You need to have augmented word dataset first and provide the path to where it is saved.
            On the fly data augmentation can be done if enough CPUs are available.
            You can download the dataset here:
            http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

        Specs:
            - 200 samples
            - Each image is (3 x 1242 x 376)

        In total there are 34 classes but some of these are not useful so by default we use only 19 of the classes
        specified by the `valid_labels` parameter.

        Example::
            For automatic split of data into train, validation and test use with appropriate val_split, test_split values
            For esisting splits pass a list of dirs then val_split, test_split values will be ignored
            
            from uitls.datamodules import PHOSCZSDataModule or PHOSCZSDataModule([train_PATH, validation_PATH, test_PATH])
            dm = PHOSCZSDataModule(PATH)
            model = LitModel()
            Trainer().fit(model, datamodule=dm)

        Args:
            data_dir: where to load the data from path, i.e. '/path/to/folder/with/data_semantics/' or ['/path/to/train_data/', '/path/to/valid_data/', '/path/to/test_data/']
            val_split: size of validation test (default 0.2)
            test_split: size of test set (default 0.1)
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)
        
        # ===== Populate train, valid,test configs from general config =====
        general_config = _dataset_configs.pop('general', None)
        dataset_configs = {}
        for key, _config in _dataset_configs.items():
            dataset_configs[key] = _config if general_config == None else recursive_parse_settings(deepcopy(general_config), _config)
        
        # ===== Preprocess the labels if required =====
        header_list = ["Image", "Word", "extra"]
        df_train, df_valid, df_test = None, None, None
        assert os.path.exists(dataset_configs['train']['label']), "Error: Train labels file does not exist"
        df_train = pd.read_csv(dataset_configs['train']['label'], names=header_list)
        
        assert os.path.exists(dataset_configs['valid']['label']), "Error: Valid labels file does not exist"
        df_valid = pd.read_csv(dataset_configs['valid']['label'], names=header_list)
        
        if 'test' in _dataset_configs.keys():
            assert os.path.exists(dataset_configs['test']['label']), "Error: Test labels file does not exist"
            df_test = pd.read_csv(dataset_configs['test']['label'], names=header_list)
        
        for df in [df_train, df_valid, df_test]:
            if df is not None and 'extra' in df.columns: # created by extra comma in csv
                df.drop(columns=['extra'], inplace=True)
        
        if dataset_configs['train']['unseen'] != None:
            assert os.path.exists(dataset_configs['train']['unseen']), "Error: Unseen train labels file does not exist"
            df_unseen = pd.read_csv(dataset_configs['train']['unseen'], names=header_list)
            df_train  = df_train.merge(df_unseen, how='left', indicator=True)
            df_train  = df_train[df_train['_merge'] == 'left_only']
            df_train  = df_train[['Image', 'Word']]
        if dataset_configs['train']['raw'] == dataset_configs['valid']['raw']:
            df_train = df_train.merge(df_valid, how='left', indicator=True)
            df_train = df_train[df_train['_merge'] == 'left_only']
            df_train = df_train[['Image', 'Word']]
        
        self.loader_config   = loader_config
        self.dataset_configs = dataset_configs
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test  = df_test
        self.df_all   = pd.DataFrame()

    def prepare_data(self):
        loader_config   = self.loader_config
        dataset_configs = self.dataset_configs
        df_train = self.df_train
        df_valid = self.df_valid
        df_test  = self.df_test

        '''print(f"Train_Images= {len(df_train)}, Valid_Images={len(df_valid)}")
        pdb.set_trace()'''
        # ===== Populate PhosLabel and PhocLabel =====
        for key, config in dataset_configs.items():
            if key == 'train':
                config['label'] = df_train
                (dataset_configs[key]['label'], 
                 train_word_phos_label, 
                 train_word_phoc_label) = self.sanity_checks(config)
                train_word_list = list(set(dataset_configs[key]['label']['Word']))
            elif key == 'valid':
                config['label'] = df_valid
                (dataset_configs[key]['label'], 
                 valid_word_phos_label, 
                 valid_word_phoc_label) = self.sanity_checks(config)
                valid_word_list = list(set(dataset_configs[key]['label']['Word']))
            elif key == 'test':
                config['label'] = df_test
                (dataset_configs[key]['label'], 
                 test_word_phos_label, 
                 test_word_phoc_label) = self.sanity_checks(config)
                test_word_list = list(set(dataset_configs[key]['label']['Word']))
                
            '''print(f"config: {config}")'''
            
        if 'test' in dataset_configs.keys():
            self.all_phos_labels = {**train_word_phos_label, **valid_word_phos_label, **test_word_phos_label}
            self.all_phoc_labels = {**train_word_phoc_label, **valid_word_phoc_label, **test_word_phoc_label}
            all_words       = list(set(train_word_list + valid_word_list + test_word_list))    
        else:
            self.all_phos_labels = {**train_word_phos_label,**valid_word_phos_label}
            self.all_phoc_labels = {**train_word_phoc_label,**valid_word_phoc_label}
            all_words       = list(set(train_word_list + valid_word_list))
        
        self.wordLabelEncoder = preprocessing.LabelEncoder()
        self.wordLabelEncoder.fit(all_words)
        
        '''
        print(f"len(train_word_phos_label): {len(train_word_phos_label)}, len(valid_word_phos_label): {len(valid_word_phos_label)}")
        print(f"len(train_word_phoc_label): {len(train_word_phoc_label)}, len(valid_word_phoc_label): {len(valid_word_phoc_label)}")
        pdb.set_trace()
        '''
        for key, config in dataset_configs.items():
            config['label']['PhosLabel'] = config['label']['Word'].apply(self.getphoslabel)
            config['label']['PhocLabel'] = config['label']['Word'].apply(self.getphoclabel)
            config['label']['Wordlabel'] = config['label']['Word'].apply(self.getwordlabel)
        
        for func, col_name in zip([self.getphoslabel, self.getphoclabel, self.getwordlabel], ['phos', 'phoc', 'word']):
            self.df_all[col_name] = list(map(func, all_words))
        
        '''print(f"config['label'].columns: {config['label'].columns}")'''
        #pdb.set_trace()
        # ===== Populate trainset, valset, testset from Dataset class =====
        self.trainset, self.valset, self.testset = None, None, None
        for key, config in dataset_configs.items():
            list_class_str = config.pop('ds_class', None)
            assert all(class_str is not None for class_str in list_class_str), 'list_class_str needs to be defined in loaders: dataset_config'
            (concat_dataset_class, 
             image_dataset_class, 
             phos_dataset_class, 
             phoc_dataset_class,
             wlabel_dataset_class) = map(get_class_from_str, list_class_str)
            '''print(f"concat_dataset_class: {concat_dataset_class}, \
                  image_dataset_class: {image_dataset_class}, \
                  phos_dataset_class: {phos_dataset_class}, \
                  phoc_dataset_class: {phoc_dataset_class}")
            
            xx1 = image_dataset_class(list(config['label']['Image']), config['transforms'])
            xx2 = phos_dataset_class(list(config['label']['PhosLabel']))
            xx3 = phoc_dataset_class(list(config['label']['PhocLabel']))
            print(f"xx1: {xx1}, \
                  xx2: {xx2}, \
                  xx3: {xx3}")'''
            dataset = concat_dataset_class({'img': image_dataset_class(list(config['label']['Image']), self._default_transforms()),
                                            'phos': phos_dataset_class(list(config['label']['PhosLabel'])), 
                                            'phoc': phoc_dataset_class(list(config['label']['PhocLabel'])),
                                            'wlabel': wlabel_dataset_class(list(config['label']['Wordlabel']))})
            if key == 'train':
                self.trainset = dataset
            elif key == 'valid':
                self.validset = dataset
            elif key == 'test':
                self.testset = dataset
                
        '''print(f"self.trainset: {self.trainset}, self.validset: {self.validset}, self.testset: {self.testset}")
        pdb.set_trace()'''
        # ===== Initialize from loader_config =====
        self.batch_size  = loader_config['batch_size']
        self.num_workers = loader_config['num_workers']
        self.seed        = loader_config['seed']
        self.shuffle     = loader_config['shuffle']
        self.pin_memory  = loader_config['pin_memory']
        self.drop_last   = loader_config['drop_last']
    
    def sanity_checks(self, dataset_config):
        data_dir  = dataset_config['raw'] if isinstance(dataset_config['raw'], list) else [dataset_config['raw']]
        labels    = dataset_config['label']
        
        # Check all images have labels corresponding to them in training
        img_list = []
        for _dir in data_dir:
            files_list = get_filenames(_dir)
            img_list   += files_list
        
        # Adding folder names to file names
        labels['Image'] = dataset_config['raw']+ "/" + labels['Image']
        
        assert all(elem in img_list  for elem in list(labels['Image'])), "Error: Not all labeled images are present"
        assert all(elem in list(labels['Image']) for elem in img_list), "Error: Not all images are labeled"
        
        # Generating dictionaries of words mapped to PHOS & PHOC vectors and len(word)
        word_phos_label = gen_phos_label(list(set(labels['Word'])))
        word_phoc_label = gen_phoc_label(list(set(labels['Word'])))
        
        #NOTE: self.labels with will be updated in the loader with PhosLabel and PhocLabel
        
        # Check all images have the same size
        IMG_DIM = np.array(Image.open(img_list[0])).shape
        assert all(IMG_DIM == np.array(Image.open(f_img)).shape for f_img in img_list), "Error: Not all images are the same shape"
        
        return labels, word_phos_label, word_phoc_label

    def getphoclabel(self, x):
        return self.all_phoc_labels[x]

    def getphoslabel(self, x):
        return self.all_phos_labels[x]
    
    def getwordlabel(self, x):
        yy = self.wordLabelEncoder.transform([x])[0]
        return yy

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def _default_transforms(self) -> Callable:
        phosc_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return phosc_transforms
