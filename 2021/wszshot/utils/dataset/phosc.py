import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import pdb

class ImageDataset(Dataset):
    """
    Note:
        You need to have augmented word dataset first and provide the path to where it is saved.
        On the fly data augmentation can be done if enough CPUs are available.
    """    
    
    def __init__(self,
                 img_list: list,
                 transfroms
                ):
        """
        Args:
            data_dir (str): path/paths to augmented data
            label_csv: CSV file containing the labels for images
            transform: transforms that need to be applied to the images
        """
        self.img_list  = img_list
        self.transform = transfroms
        
        
    def __len__(self):
        return len(self.img_list)
    
    
    def get_batch_images(self, idx):
        # Fetch a batch of inputs        
        img = np.array(Image.open(self.img_list[idx]))
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __getitem__(self, idx):
        batch_img = self.get_batch_images(idx)
        return batch_img
    

class PhosDataset(Dataset):
    """
    Note:
        You need to have augmented word dataset first and provide the path to where it is saved.
        On the fly data augmentation can be done if enough CPUs are available.
    """    
    
    def __init__(self,
                 phos_list: list
                ):
        """
        Args:
            data_dir (str): path/paths to augmented data
            label_csv: CSV file containing the labels for images
            transform: transforms that need to be applied to the images
        """
        self.phos_list = phos_list
        
    def __len__(self):
        return len(self.phos_list)

    def __getitem__(self, idx):
        batch_phos = self.phos_list[idx]
        return batch_phos


class PhocDataset(Dataset):
    """
    Note:
        You need to have augmented word dataset first and provide the path to where it is saved.
        On the fly data augmentation can be done if enough CPUs are available.
    """    
    
    def __init__(self,
                 phoc_list: list
                ):
        """
        Args:
            data_dir (str): path/paths to augmented data
            label_csv: CSV file containing the labels for images
            transform: transforms that need to be applied to the images
        """
        self.phoc_list = phoc_list
        
    def __len__(self):
        return len(self.phoc_list)

    def __getitem__(self, idx):
        batch_phoc = self.phoc_list[idx]
        return batch_phoc


class PHOSCZSDataset(Dataset): # ConcatDataset
    def __init__(self, dict_datasets):
        self.dict_datasets = dict_datasets

    def __len__(self):
        return min(len(dataset) for dataset in self.dict_datasets.values()) # Taken from df so all lenghts will be equal
    
    def __getitem__(self, idx):
        batch_phosc = {}
        for key, dataset in self.dict_datasets.items():
            batch_phosc[key] = dataset[idx]
        return batch_phosc
