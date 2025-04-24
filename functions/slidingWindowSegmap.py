"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504241146
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from skimage import morphology

from ..common.utils import log_print, highlight, highlight_show
from ..processors.img_processor import ImgProcessor
from ..models.resnet101_IMAGENET1K_V2.modeling_resnet import FastResModel


def load_model( # ima only res loading & num_classes=3...
    model_path: str,
    device='cuda',
):
    log_print('load_model', f'loading model from {model_path}')
    model = FastResModel(num_classes=3)
    checkpoint = torch.load(
        model_path, 
        map_location=device
    )
    msg = model.load_state_dict(checkpoint['whold_model'].state_dict())
    model.eval()
    model = model.to(device)
    log_print('load_model', f'msg {msg}')

    return model

def calculate_window_positions(
    image_width: int, 
    image_height: int, 
    window_size: int, 
    min_overlap: int, 
):
    if min_overlap >= window_size:
        raise ValueError("min_overlap must be less than window_size.")
    stride = window_size - min_overlap

    num_windows_w = math.ceil((image_width - min_overlap) / stride)
    num_windows_h = math.ceil((image_height - min_overlap) / stride)
    
    stride_w = (image_width - window_size) / max(1, num_windows_w - 1) if num_windows_w > 1 else 0
    stride_h = (image_height - window_size) / max(1, num_windows_h - 1) if num_windows_h > 1 else 0
    
    window_positions = []
    
    if image_width <= window_size and image_height <= window_size:
        return [(0, 0)]
    
    # image_width
    if image_width <= window_size:
        x_positions = [0]
    else:
        x_positions = [int(i * stride_w) for i in range(num_windows_w)]
        if x_positions and x_positions[-1] + window_size > image_width:
            x_positions[-1] = image_width - window_size
    
    # image_height
    if image_height <= window_size:
        y_positions = [0]
    else:
        y_positions = [int(i * stride_h) for i in range(num_windows_h)]
        if y_positions and y_positions[-1] + window_size > image_height:
            y_positions[-1] = image_height - window_size
    
    for y in y_positions:
        for x in x_positions:
            window_positions.append((x, y))
    
    return window_positions

def sliding_window_detection(
    model: nn.Module, 
    image_path: str, 
    window_size:int =48, 
    min_overlap:int =24, 
    confidence_threshold:float =0.5,
    device='cuda'
):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    height, width, _ = image_np.shape
    segmentation_map = np.zeros((height, width), dtype=np.uint8)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    
    transform = ImgProcessor(img_size=window_size, macenko_nor=False, device=device)
    window_positions = calculate_window_positions(width, height, window_size, min_overlap)
    
    window_predictions = []
    model = model.to(device)
    
    for x, y in window_positions:
        end_x = min(x + window_size, width)
        end_y = min(y + window_size, height)
        actual_width = end_x - x
        actual_height = end_y - y
        
        window = image.crop((x, y, end_x, end_y))
        
        if actual_width != window_size or actual_height != window_size:
            window = window.resize((window_size, window_size))
        
        input_tensor = transform(window).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        confidence, predicted_class = torch.max(probabilities, 0)
        confidence = confidence.item()
        predicted_class = predicted_class.item()
        if confidence >= confidence_threshold:
            final_class = predicted_class + 1
        else:
            final_class = 0

        window_predictions.append({
            'x': x,
            'y': y,
            'width': actual_width,
            'height': actual_height,
            'predicted_class': final_class,
            'confidence': confidence
        })
        
        for i in range(y, end_y):
            for j in range(x, end_x):
                segmentation_map[i, j] = final_class
                confidence_map[i, j] = confidence
    
    return image_np, segmentation_map, confidence_map, window_predictions

def visualize_results(
    image, 
    segmentation_map, 
    window_predictions
):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(image)
    for pred in window_predictions:
        rect = patches.Rectangle(
            (pred['x'], pred['y']), 
            pred['width'], 
            pred['height'], 
            linewidth=1, 
            edgecolor='r' if pred['predicted_class'] == 1 else 'b' if pred['predicted_class'] == 2 else 'g',
            facecolor='none'
        )
        axs[0, 1].add_patch(rect)
    axs[0, 1].set_title('window_predictions')
    axs[0, 1].axis('off')
    
    cmap = ListedColormap(['black', 'red', 'blue', 'green'])
    axs[1, 0].imshow(segmentation_map, cmap=cmap, vmin=0, vmax=3)
    axs[1, 0].set_title('segmentation_map')
    axs[1, 0].axis('off')
    
    overlay = np.copy(image)
    mask_colors = np.array([
        [0, 0, 0, 0], 
        [255, 0, 0, 128], 
        [0, 0, 255, 128], 
        [0, 255, 0, 128]
    ], dtype=np.uint8)
    
    overlay_mask = np.zeros((*segmentation_map.shape, 4), dtype=np.uint8)
    for i in range(1, 4):
        overlay_mask[segmentation_map == i] = mask_colors[i]
    
    alpha = overlay_mask[..., 3:4] / 255.0
    overlay = overlay * (1 - alpha) + overlay_mask[..., :3] * alpha
    overlay = overlay.astype(np.uint8)
    
    axs[1, 1].imshow(overlay)
    axs[1, 1].set_title('overlay')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    legend_elements = [
        patches.Patch(facecolor='red', alpha=0.5, label='class 0'),
        patches.Patch(facecolor='blue', alpha=0.5, label='class 1'),
        patches.Patch(facecolor='green', alpha=0.5, label='class 2'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.show()

def get_bin_seg_map(seg_map):
    only_result = np.zeros(seg_map.shape, dtype=np.uint8)
    only_result[seg_map == 1] = 1
    only_result[seg_map == 2] = 1

    only_result = morphology.remove_small_holes(only_result, 8888)
    only_result = morphology.remove_small_objects(only_result, 88888)

    bin_seg_map = np.zeros(seg_map.shape, dtype=np.uint8)
    bin_seg_map[only_result == True] = 1

    return bin_seg_map











