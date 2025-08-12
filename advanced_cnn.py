import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import logging
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split

# logging setup

logging.basicConfig(level = logging.INFO , format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   
                   handlers = [logging.StreamHandler(), logging.FileHandler('deepfake_detection.log')]
                    )

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.data_dir = "./deepfake_dataset"
        self.classes = ['real','deepfake']
        self.train_split = 0.7
        self.val_split = 0.2
        self.test_split = 0.1
        self.batch_size = 2
        self.patch_size = 16
        self.img_size = 224
        self.accumulation_steps = 2
        self.epochs = 25
        self.attention_heads = 16
        self.model_path ='deepfake_vit_model.pt'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr= 0.0001
        self.max_lr = 0.0015
        self.weight_decay = 1e-4 
        self.dropout_rate = 0.2
        logger.info(f"Using device {self.device}")
        
        
        