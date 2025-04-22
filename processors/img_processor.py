"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504221440
"""

import cv2
import numpy as np
import random
import albumentations as A
from torchvision import transforms
import torch.nn.functional as F
import torch


class ImgProcessor():
    def __init__(self, 
        img_size:int =None, 
        macenko_nor:bool =False, 
        device:str ='cuda'
    ):
        self.device = device
        self.img_size = img_size
        self.macenko_nor = macenko_nor

        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.5),

            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),

            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.1), 

            # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.1), 

            # A.CoarseDropout(max_holes=3, max_height=20, max_width=20, fill_value=random.randint(100, 200), p=0.1)
        ])
        self.pre_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.707223, 0.578729, 0.703617), 
            #     std=(0.211883, 0.230117, 0.177517)
            # ),
        ])
        
    @staticmethod
    def macenko_normalization(
        I, 
        Io=240, 
        alpha=1, 
        beta=0.15, 
        target_max=None
    ):
        I = I.astype(np.float32)
        OD = -np.log((I + 1) / Io)
        
        mask = (I < Io).all(axis=2)
        OD_hat = OD[mask].reshape(-1, 3)
        OD_hat = OD_hat[np.max(OD_hat, axis=1) > beta]
        
        if OD_hat.shape[0] == 0:
            return I.astype(np.uint8)
        
        U, S, V = np.linalg.svd(OD_hat, full_matrices=False)
        V = V[:2, :].T  # 形狀 (3, 2)
        
        That = np.dot(OD_hat, V)
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        stain_matrix = np.array([v1, v2]).T
        stain_matrix /= np.linalg.norm(stain_matrix, axis=0)
        
        OD_flat = OD.reshape((-1, 3)).T
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, OD_flat, rcond=None)
        
        maxC = np.percentile(concentrations, 99, axis=1)
        if target_max is None:
            target_max = maxC
        
        norm_concentrations = concentrations * (target_max[:, None] / maxC[:, None])
        OD_normalized = np.dot(stain_matrix, norm_concentrations)
        
        I_normalized = Io * np.exp(-OD_normalized)
        I_normalized = I_normalized.T.reshape(I.shape)
        I_normalized = np.clip(I_normalized, 0, 255).astype(np.uint8)
        
        return I_normalized
    
    def __call__(
        self, 
        image
    ):
        image_np = np.array(image)
        aug_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # augmentation
        if self.macenko_nor:
            aug_image = self.macenko_normalization(aug_image)
        aug_image = self.augmentation_pipeline(image=aug_image)
        aug_image = transforms.ToPILImage()(aug_image['image'])

        # to tensor
        input_image = self.pre_transform(aug_image).to(self.device) # .unsqueeze(0)
        if self.img_size is not None:
            input_image = F.interpolate(input_image.unsqueeze(0), size=(self.img_size, self.img_size), mode='bicubic', align_corners=False).squeeze(0)

        return input_image
    
    @classmethod
    def from_config(cls, 
        cfg,
        img_size:int =None,
    ):
        # root_path = cfg['task'].get("root_path")
        device = str(cfg['task'].get("device"))

        if cfg['dataset'].get('img_processor') is not None:
            img_processor_cfg = cfg['dataset']['img_processor']
            macenko_nor = bool(img_processor_cfg.get('macenko_nor'))

        processor = cls(
            img_size=img_size, 
            macenko_nor=macenko_nor, 
            device=device
        )
        return processor


def get_strong_transform(
    img_size: int,
):
    train_transform = transforms.Compose([
        transforms.Resize((550, 550), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(img_size), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform

def tta_predict(
    model, 
    images,
    img_size: int,
):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        outputs = model(images)
        predictions.append(outputs)
    
    with torch.no_grad():
        flipped = torch.flip(images, [3])
        outputs = model(flipped)
        predictions.append(outputs)
    
    with torch.no_grad():
        flipped = torch.flip(images, [2])
        outputs = model(flipped)
        predictions.append(outputs)
    
    with torch.no_grad():
        rotated = torch.rot90(images, 1, [2, 3])
        outputs = model(rotated)
        predictions.append(outputs)
    
    # with torch.no_grad():
    #     center_crop = transforms.CenterCrop(img_size)(images)
    #     outputs = model(center_crop)
    #     predictions.append(outputs)
    
    final_pred = torch.stack(predictions).mean(0)
    return final_pred












