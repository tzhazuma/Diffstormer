from typing import Sequence, Dict, Union
import math
import time
import random
import SimpleITK as sitk
import warnings
import torchio as tio
import sys
import os
import imageio
#sys.path.append("/home_data/home/lifeng2023/code/moco/DiffBIR-main")
sys.path.append("/public_bme/data/lifeng/code/moco/MoCo-Diff")
import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
from einops import rearrange
from ..utils.file import load_file_list
from ..utils.image import center_crop_arr, augment, random_crop_arr, center_pad_arr,center_resize_arr
from ..utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)
import torch
from dataset import motion_sim
import torch.nn.functional as F

warnings.filterwarnings('ignore')
reader = sitk.ImageSeriesReader()

def load_dcm(folder_path, error_ids=[]):
    try:
        dicomFiles = reader.GetGDCMSeriesFileNames(folder_path)
        reader.SetFileNames(dicomFiles)
        vol = reader.Execute()
    except:
        error_ids.append(folder_path.split('/')[-1])
        return None
    image_array = sitk.GetArrayFromImage(vol)
    return image_array

def load_nii(nii_path):
    try:
        image = sitk.ReadImage(nii_path)
    except:
        assert False, f"failed to load image {nii_path}"
    image_array = sitk.GetArrayFromImage(image)
    
    return image_array

def save_nii(array, id_name):
    img = sitk.GetImageFromArray(array)
    sitk.WriteImage(img, '/public_bme/data/lifeng/data/LISA/LISA_train_motion/' + id_name)

def normalize_mri_volume(volume):
    """
    Normalize the MRI 3D volume by clipping the intensity values to the 0.3%-99.7% range
    and setting values above 99.7% to the maximum value in the clipped range.

    Parameters:
    volume (numpy.ndarray): The input 3D MRI volume.

    Returns:
    numpy.ndarray: The normalized 3D MRI volume.
    """
    # Compute the 0.3% and 99.7% intensity values
    lower_bound = np.percentile(volume, 0.3)
    upper_bound = np.percentile(volume, 99.7)
    
    # # Clip the intensity values to the 0.3%-99.7% range
    # clipped_volume = np.clip(volume, lower_bound, upper_bound)
    
    # Set values above 99.7% to the maximum value in the clipped range
    volume[volume > upper_bound] = upper_bound
    volume[volume < lower_bound] = lower_bound
    
    return volume#.astype("float32")

class MotionbrainDataset_2(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        # crop_type: str,
        # use_hflip: bool
    ) -> "MotionbrainDataset_2":
        super(MotionbrainDataset_2, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)[0:20]
    
        self.out_size = out_size       #512
        # self.crop_type = crop_type     #center
        # assert self.crop_type in ["none", "center", "random"]   
        # self.use_hflip = use_hflip
        self.target_all = []
        self.source_all = []
        self.num = 1


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        
        gt_path = self.paths[index]
 
        success = False
        try:
            pil_img_3D = load_nii(gt_path)
            success = True
        except:
            return self.__getitem__(index+1)
        assert success, f"failed to load image {gt_path}"
        
            
        # ------------------------ generate lq image ------------------------ #
        corruption_scheme = ['piecewise_constant', 'gaussian', 'piecewise_transient'] 
        selected_corruption_scheme = np.random.choice(corruption_scheme)
        # # ------------------------ generate lq image ------------------------ #
        try:
            ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[30, 40],
                                                corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
                                                n_seg=8)
            #transformed_img_3D = torch.tensor(ms_layer.layer_op(pil_img_3D))
            transformed_img_3D = ms_layer.layer_op(pil_img_3D)

        except:
            ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[30, 30],
                                                corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
                                                n_seg=8)
            transformed_img_3D = ms_layer.layer_op(pil_img_3D)

        length = pil_img_3D.shape[0]
        median = round(length / 2)
        # for index_A in range(74,84):
        index_A = random.randint(median-40, median+50)


        transformed_img_3D = normalize_mri_volume(transformed_img_3D)
        max_value, min_value = transformed_img_3D.max(), transformed_img_3D.min() 
        transformed_img_3D = (transformed_img_3D - min_value) / (max_value - min_value)

        try:
            
            pil_img = pil_img_3D[index_A-2:index_A+4]
            lq_img = transformed_img_3D[index_A-2:index_A+4]
            
        except:
            print(f"failed to load image {gt_path}")

            index_A = random.randint(90, 130)          
            lq_img = transformed_img_3D[index_A-2:index_A+4]
            pil_img = pil_img_3D[index_A-2:index_A+4]

        pil_img = center_crop_arr(np.array(pil_img), self.out_size, dim=3)
        lq_img = center_crop_arr(np.array(lq_img), self.out_size, dim=3)

        """Normalize img to [0,1]
        """
        max_value, min_value = lq_img.max(), lq_img.min() 
        lq_img = (lq_img - min_value) / (max_value - min_value)

        max_value, min_value = img_gt.max(), img_gt.min() 
        img_gt = (img_gt - min_value) / (max_value - min_value)

        img_gt = np.stack((img_gt[0], img_gt[0], img_gt[0]), axis=-1)
        lq_img = np.stack((lq_img, lq_img, lq_img), axis=-1)
            
        img_gt = np.array(pil_img)
        lq_img = np.array(lq_img)

        target = (img_gt * 2 - 1).astype(np.float32)

        source = lq_img.astype(np.float32)


        # data = (img_gt * 255.0).astype('uint8')  # 转换数据类型
        # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
        # # # 保存图片到本地
        # imageio.imsave(f'/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/{index}_{index_A}_hq.jpg', new_im_hq)
        # data = (source * 255.0).astype('uint8')  # 转换数据类型
        # new_im_lq = Image.fromarray(data)  # 调用Image库，数组归一化
        # # # # 保存图片到本地
        # imageio.imsave(f'/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/{index}_{index_A}_lq.jpg', new_im_lq)
            
        return dict(jpg=target, txt="", hint=source)

    def __len__(self) -> int:
        return len(self.paths)

if __name__ == '__main__':
    data = MotionbrainDataset_2(
        file_list= '/public_bme/data/lifeng/data/IXI/T1_brain', #'/public_bme/data/lifeng/data/LISA/LISA2024Task1', #'/public_bme/data/lifeng/data/val_hcp.list',
        out_size = 512,
        # crop_type= 'center',
        # #crop_type= 'none',
        # use_hflip=False
        )
    for i in data:
       pass
