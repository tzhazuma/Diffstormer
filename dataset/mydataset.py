from PIL import Image
from torch.utils.data import Dataset
import os
from os import path

class MyDataSet(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir_ori = path.join(img_dir,"ori")
        self.img_dir_ref = path.join(img_dir,"ref")
        self.transform = transform
        self.imgs_ori = os.listdir(self.img_dir_ori)
        self.imgs_ref = os.listdir(self.img_dir_ref)
    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, idx):
        img_name_ori = os.path.join(self.img_dir_ori, self.imgs_ori[idx])
        image_ori = Image.open(img_name_ori)
        img_name_ref = os.path.join(self.img_dir_ref, self.imgs_ref[idx])
        image_ref = Image.open(img_name_ref)
        if self.transform:
            image_ori = self.transform(image_ori)
            image_ref = self.transform(image_ref)
        return image_ori,image_ref