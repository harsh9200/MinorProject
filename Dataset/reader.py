import os
import cv2
import numpy as np
import pandas as pd

from utils import run_length_decode
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import ShiftScaleRotate, Normalize, Resize, Compose



def prepare_dataloader(
    RLE_DF_PATH,
    base_path,
    phase,
    fold=4,
    total_folds=5,
    batch_size=16,
    num_workers=4,
    ):
    RLE_DF_ALL = pd.read_csv(RLE_DF_PATH)
    
    #? Rename Column Names
    RLE_DF_ALL.columns = ['ImageId', 'EncodedPixels']
    
    #? Drop duplicate values in ImageId column
    RLE_DF = RLE_DF_ALL.drop_duplicates('ImageId')
    
    RLE_DF_WITH_MASK = RLE_DF[RLE_DF["EncodedPixels"] != " -1"]
    RLE_DF_WITH_MASK['has_mask'] = 1
    
    RLE_DF_WITHOUT_MASK = RLE_DF[RLE_DF["EncodedPixels"] == " -1"]
    RLE_DF_WITHOUT_MASK['has_mask'] = 0
    
    #? Sample equal number of masked and non-masked images
    RLE_DF_WITHOUT_MASK_SAMPLED = RLE_DF_WITHOUT_MASK.sample(len(RLE_DF_WITH_MASK), random_state=69)
    
    RLE_DF = pd.concat([RLE_DF_WITH_MASK, RLE_DF_WITHOUT_MASK_SAMPLED])
    
    
    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(RLE_DF["ImageId"], RLE_DF["has_mask"]))[fold]
    Train_DF, Val_DF = RLE_DF.iloc[train_idx], RLE_DF.iloc[val_idx]
    DF = Train_DF if phase == "train" else Val_DF
    
    
    File_Names = DF['ImageId'].values
    
    image_dataset = SIIMDataset(RLE_DF_ALL, File_Names, base_path)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


class SIIMDataset(Dataset):
    def __init__(self, RLE_DF, File_Names, Base_Path):
        self.RLE_DF = RLE_DF
        self.BASE_PATH = Base_Path
        self.File_Names =  File_Names
        self.RLE_DF_GroupBy = self.RLE_DF.groupby('ImageId')
        
        self.Transforms = Compose([
            ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            Resize(512, 512),
            Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225), 
                p=1
            ),
            ToTensorV2(),
        ])
    
    def __getitem__(self, idx):
        img_id = self.File_Names[idx]
        DF = self.RLE_DF_GroupBy.get_group(img_id)
        Annotations = DF['EncodedPixels'].tolist()
        img_path = os.path.join(self.BASE_PATH, img_id + '.png')
        
        img = cv2.imread(img_path)
        Mask = np.zeros([1024, 1024])
        print(Annotations)
        if Annotations[0] != ' -1':
            for rle in Annotations:
                Mask += run_length_decode(rle)
        
        Mask = (Mask >= 1).astype('float32')
        Augmented = self.Transforms(image=img, mask=Mask)
        
        img, Mask = Augmented['image'], Augmented['mask']
        
        return img, Mask
    
    def __len__(self):
        return len(self.File_Names)
