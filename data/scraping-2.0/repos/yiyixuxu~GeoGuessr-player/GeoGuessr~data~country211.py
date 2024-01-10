## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

from math import ceil
import pytorch_lightning as pl

from torchvision.datasets import Country211 as torch_Country211
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from typing import Callable 
import ml_collections
from pathlib import Path
import os


from .base import BaseDataModule, GPSBaseDataset
from .util import get_files

class Country211(BaseDataModule):
    """The Country211 Data Set 
     <https://github.com/openai/CLIP/blob/main/data/country211.md>_ from OpenAI.

    This dataset was built by filtering the images from the YFCC100m dataset
    that have GPS coordinate corresponding to a ISO-3166 country code. The
    dataset is balanced by sampling 150 train images, 50 validation images, and
    100 test images images for each country.
    """

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__(config)
        self.transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
    def prepare_data(self):
        if not os.path.exists(self.data_dir / self.dataset_name):
            torch_Country211(self.data_dir, download = True)
    
    def setup(self, stage: str=None) -> None:
        if stage == 'fit' or stage is None:
            self.data_train = torch_Country211(self.data_dir, split = 'train', transform = self.transform)
            self.data_val = torch_Country211(self.data_dir, split = 'valid',transform = self.transform)
            self.class_to_idx = self.data_train.class_to_idx
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        if stage == "test" or stage is None:
            self.data_test = torch_Country211(self.data_dir, split='test',transform = self.transform)


def FiveCrop_tranform(image):
    image = transforms.Resize(256)(image)
    crops = transforms.FiveCrop(224)(image)
    crops_transformed = []
    crop_transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    
    for crop in crops:
        crops_transformed.append(crop_transform(crop))
    return torch.stack(crops_transformed, dim=0)



class Country211_GeoEstimation(BaseDataModule):
    """The Country211 Data Set 
     <https://github.com/openai/CLIP/blob/main/data/country211.md>_ from OpenAI.

    This dataset was built by filtering the images from the YFCC100m dataset
    that have GPS coordinate corresponding to a ISO-3166 country code. The
    dataset is balanced by sampling 150 train images, 50 validation images, and
    100 test images images for each country.
    """

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__(config)
        self.transform = FiveCrop_tranform
    

    def parepare_data(self):
        if not os.path.exists(self.data_dir / self.dataset_name):
            torch_Country211(self.data_dir, download = True)
 
    def setup(self, stage: str=None ) -> None:
  
        if stage == 'fit' or stage is None:
            img_paths_train = get_files(self.data_dir / self.dataset_name, extensions = self.extensions, folders=['train'])
            img_paths_val = get_files(self.data_dir / self.dataset_name, extensions = self.extensions, folders=['valid'])

            self.data_train = GPSBaseDataset(img_paths_train,transform = self.transform)
            self.data_val = GPSBaseDataset(img_paths_val,transform = self.transform)
        
        if stage == "test" or stage is None:
            img_paths_test = get_files(self.data_dir / self.dataset_name, extensions = self.extensions, folders=['test'])

            self.data_test = GPSBaseDataset(img_paths_test,transform = self.transform)
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=ceil(self.batch_size/5),
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
