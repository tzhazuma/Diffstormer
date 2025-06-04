import torch
import torchvision as tv
import torchvision.models as models
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2 as cv
import sklearn as sk
import os

def get_resnet(model_name, pretrained=False, num_classes=1000):
    """
    获取torchvision中的resnet或resnext模型

    Args:
        model_name (str): 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                          'resnext50_32x4d', 'resnext101_32x8d'
        pretrained (bool): 是否加载预训练权重
        num_classes (int): 分类数

    Returns:
        nn.Module: resnet或resnext模型
    """
    resnet_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'resnext50_32x4d': models.resnext50_32x4d,
        'resnext101_32x8d': models.resnext101_32x8d,
    }
    if model_name not in resnet_dict:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = resnet_dict[model_name](pretrained=pretrained)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
