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
from skimage import segmentation, color, graph
from typing import Tuple, List, Dict
import cv2

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

def sliding_window_detection_(
    model: nn.Module, 
    image_path: str, 
    window_size:int =48, 
    min_overlap:int =24, 
    conf_thr:float =0.5,
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
        
        input_tensor = transform.testing(window).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        confidence, predicted_class = torch.max(probabilities, 0)
        confidence = confidence.item()
        predicted_class = predicted_class.item()
        if confidence >= conf_thr:
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

def sliding_window_detection(
    model: nn.Module,
    image_path: str,
    window_size: int = 48,
    min_overlap: int = 24,
    conf_thr: float = 0.5,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    height, width, _ = image_np.shape
    segmentation_map = np.zeros((height, width), dtype=np.uint8)
    confidence_map   = np.zeros((height, width), dtype=np.float32)

    transform = ImgProcessor(img_size=window_size, macenko_nor=False, device=device)
    window_positions = calculate_window_positions(width, height, window_size, min_overlap)
    model = model.to(device).eval()

    tensors: List[torch.Tensor] = []
    metas:    List[Tuple[int,int,int,int]] = []
    for x, y in window_positions:
        end_x = min(x + window_size, width)
        end_y = min(y + window_size, height)
        actual_w = end_x - x
        actual_h = end_y - y

        win = image.crop((x, y, end_x, end_y))
        if actual_w != window_size or actual_h != window_size:
            win = win.resize((window_size, window_size))

        t = transform.testing(win).unsqueeze(0).to(device)  # (1, C, H, W)
        tensors.append(t)
        metas.append((x, y, actual_w, actual_h))

    if not tensors:
        return image_np, segmentation_map, confidence_map, []

    batch = torch.cat(tensors, dim=0)  # (N, C, H, W)
    with torch.no_grad():
        logits = model(batch)                # (N, num_classes)
        probs  = torch.softmax(logits, dim=1)# (N, num_classes)

    window_predictions = []
    for i, (x, y, w, h) in enumerate(metas):
        prob = probs[i]
        confidence, cls0 = prob.max(dim=0)
        confidence = confidence.item()
        predicted_class = cls0.item() + 1 if confidence >= conf_thr else 0

        window_predictions.append({
            'x': x, 'y': y,
            'width': w, 'height': h,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

        segmentation_map[y : y + h, x : x + w] = predicted_class
        confidence_map  [y : y + h, x : x + w] = confidence

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

def crop_binary_array(image_array, binary_mask):
    rows, cols = np.where(binary_mask == 1)
    if len(rows) == 0:
        return image_array, binary_mask
    
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    cropped_image = image_array[min_row:max_row+1, min_col:max_col+1]
    cropped_mask = binary_mask[min_row:max_row+1, min_col:max_col+1]
    return cropped_image, cropped_mask

def _weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (
        graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count']
    )

def region_adjacency_graph(
    image:np.ndarray,
    n_segments:int=80
):
    labels = segmentation.slic(image, compactness=30, n_segments=n_segments, start_label=1)
    g = graph.rag_mean_color(image, labels)

    labels2 = graph.merge_hierarchical(
        labels,
        g,
        thresh=35,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_mean_color,
        weight_func=_weight_mean_color,
    )
    out = color.label2rgb(labels2, image, kind='avg', bg_label=0)
    # out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

    return out, labels2

def mask_otherRect_byPoint(
    image:np.ndarray,
    mask:np.ndarray,
    class_map:np.ndarray,
    MaskRect_ratio_thr:float=0.75,
    LocalgGobal_ratio_thr:float=0.50,
    color=(255, 255, 255)
):
    result = image.copy()
    mask_global_count = np.sum(np.isin(mask, 1))
    for pixel_idx in np.unique(class_map).tolist():
        single_class = np.zeros(class_map.shape, dtype=np.uint8)
        single_class[class_map == pixel_idx] = 1
        
        contours, _ = cv2.findContours(single_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.002 * perimeter, False)

            if 4 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(approx)

                cropped_mask = mask[y:y+h+1, x:x+w+1]
                mask_local_count = np.sum(np.isin(cropped_mask, 1))
                MaskRect_ratio = mask_local_count / float(w * h)
                LocalgGobal_ratio = mask_local_count / mask_global_count
                if MaskRect_ratio >= MaskRect_ratio_thr or LocalgGobal_ratio >= LocalgGobal_ratio_thr:
                    continue
            
                area_ratio = cv2.contourArea(contour) / (w * h)
                if area_ratio > 0.8:
                    cv2.rectangle(result, (x, y), (x+w, y+h), color, -1)
    return result

def mask_otherRect_byArea(
    image:np.ndarray,
    mask:np.ndarray,
    class_map:np.ndarray,
    area_ratio_thr:float=0.8,
    MaskRect_ratio_thr:float=0.75,
    LocalgGobal_ratio_thr:float=0.50,
    color=(255, 255, 255)
):
    result = image.copy()
    mask_global_count = np.sum(np.isin(mask, 1))
    for pixel_idx in np.unique(class_map).tolist():
        single_class = np.zeros(class_map.shape, dtype=np.uint8)
        single_class[class_map == pixel_idx] = 1
        
        contours, _ = cv2.findContours(single_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue
            
            cropped_mask = mask[y:y+h+1, x:x+w+1]
            mask_local_count = np.sum(np.isin(cropped_mask, 1))
            MaskRect_ratio = mask_local_count / float(w * h)
            LocalgGobal_ratio = mask_local_count / mask_global_count
            if MaskRect_ratio >= MaskRect_ratio_thr or LocalgGobal_ratio >= LocalgGobal_ratio_thr:
                continue

            area_ratio = cv2.contourArea(cnt) / float(w * h)
            if area_ratio > area_ratio_thr and not (w == image.shape[1] and h == image.shape[0]):
                cv2.rectangle(result, (x, y), (x + w, y + h), color, -1)
    return result

def remove_border(
    image:np.ndarray,
    mask:np.ndarray,
    class_map:np.ndarray,
    mask_ratio:float=0.5
):
    if class_map.size == 0:
        return image, mask, class_map
    h, w = class_map.shape
    
    non_uniform_rows = []
    for i in range(h):
        row = class_map[i]
        mask_row = mask[i]
        if (not np.all(row == row[0])) or float(np.sum(np.isin(mask_row, 1))) / len(mask_row.tolist()) > mask_ratio:
            non_uniform_rows.append(i)
    
    non_uniform_cols = []
    for j in range(w):
        col = class_map[:, j]
        mask_col = mask[:, j]
        if (not np.all(col == col[0])) or float(np.sum(np.isin(mask_col, 1))) / len(mask_col.tolist()) > mask_ratio:
            non_uniform_cols.append(j)
    
    if not non_uniform_rows or not non_uniform_cols:
        return image, mask, class_map
    
    top = min(non_uniform_rows)
    bottom = max(non_uniform_rows)
    left = min(non_uniform_cols)
    right = max(non_uniform_cols)

    cropped_image = image[top:bottom+1, left:right+1]
    cropped_mask = mask[top:bottom+1, left:right+1]
    class_map = class_map[top:bottom+1, left:right+1]
    
    return cropped_image, cropped_mask, class_map

class imgCleaner():
    def __init__(self,
        model_path:str,
        SWD_window_size:int = 48,
        SWD_min_overlap:int = 24,
        SWD_conf_thr:float = 0.05,
        RAGl1_n_segments:int = 80,
        RBl1_mask_ratio:float = 0.75,
        MOA_MaskRect_ratio_thr:float = 0.75,
        MOA_LocalgGobal_ratio_thr:float = 0.50,
        MOA_area_ratio_thr:float = 0.8,
        RAGl2_n_segments:int = 30,
        RBl2_mask_ratio:float = 0.75,
        MOP_MaskRect_ratio_thr:float = 0.75,
        MOP_LocalgGobal_ratio_thr:float = 0.50,
        device:str = "cuda",
    ):
        self.state_name = 'imgCleaner'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        model = load_model(
            model_path=model_path
        )
        self.model = model.to(self.device)
        self.SWD_window_size = SWD_window_size
        self.SWD_min_overlap = SWD_min_overlap
        self.SWD_conf_thr = SWD_conf_thr
        self.RAGl1_n_segments = RAGl1_n_segments
        self.RBl1_mask_ratio = RBl1_mask_ratio
        self.MOA_MaskRect_ratio_thr = MOA_MaskRect_ratio_thr
        self.MOA_LocalgGobal_ratio_thr = MOA_LocalgGobal_ratio_thr
        self.MOA_area_ratio_thr = MOA_area_ratio_thr
        self.RAGl2_n_segments = RAGl2_n_segments
        self.RBl2_mask_ratio = RBl2_mask_ratio
        self.MOP_MaskRect_ratio_thr = MOP_MaskRect_ratio_thr
        self.MOP_LocalgGobal_ratio_thr = MOP_LocalgGobal_ratio_thr

        log_print(self.state_name, f"...Done\n")

    def __call__(self,
        image_path:str,
    ):
        image_v0, segmentation_map, _, _ = sliding_window_detection(
            model=self.model, 
            image_path=image_path, 
            window_size=self.SWD_window_size, 
            min_overlap=self.SWD_min_overlap, 
            conf_thr=self.SWD_conf_thr, 
            device=self.device
        )
        seg_map_v0 = get_bin_seg_map(
            seg_map=segmentation_map
        )
        if np.array_equal(np.unique(seg_map_v0), [0]):
            log_print(self.state_name, "Skipping all nan array")
            return seg_map_v0
        
        cropped_image_v1, cropped_mask_v1 = crop_binary_array(
            image_array=image_v0,
            binary_mask=seg_map_v0,
        )
        
        _, class_map_v1 = region_adjacency_graph(
            image=cropped_image_v1, 
            n_segments=self.RAGl1_n_segments
        )
        cropped_image_v2, cropped_mask_v2, class_map_v2 = remove_border(
            image=cropped_image_v1, 
            mask=cropped_mask_v1, 
            class_map=class_map_v1,
            mask_ratio=self.RBl1_mask_ratio
        )

        masked_v2 = mask_otherRect_byArea(
            image=cropped_image_v2, 
            mask=cropped_mask_v2,
            class_map=class_map_v2,
            MaskRect_ratio_thr=self.MOA_MaskRect_ratio_thr,
            LocalgGobal_ratio_thr=self.MOA_LocalgGobal_ratio_thr,
            area_ratio_thr=self.MOA_area_ratio_thr,
            # color=(255, 0, 0)
        )

        _, class_map_v2 = region_adjacency_graph(
            image=masked_v2, 
            n_segments=self.RAGl2_n_segments
        )
        cropped_masked_v3, cropped_mask_v3, class_map_v3 = remove_border(
            image=masked_v2, 
            mask=cropped_mask_v2, 
            class_map=class_map_v2,
            mask_ratio=self.RBl2_mask_ratio
        )

        masked_v3 = mask_otherRect_byPoint(
            image=cropped_masked_v3, 
            mask=cropped_mask_v3,
            class_map=class_map_v3,
            MaskRect_ratio_thr=self.MOP_MaskRect_ratio_thr,
            LocalgGobal_ratio_thr=self.MOP_LocalgGobal_ratio_thr,
            # color=(0, 255, 0)
        )

        return masked_v3

    @classmethod
    def from_config(cls, 
        cfg, 
    ):
        if cfg.get("task") is not None:
            task_cfg = cfg['task']
            device = str(task_cfg.get("device"))

        if cfg.get("model") is not None:
            model_cfg = cfg['model']
            model_path = str(model_cfg.get("model_path"))


        if cfg['process'].get("sliding_window_detection") is not None:
            SWD_cfg = cfg['process']['sliding_window_detection']
            SWD_window_size = int(SWD_cfg.get("window_size"))
            SWD_min_overlap = int(SWD_cfg.get("min_overlap"))
            SWD_conf_thr = float(SWD_cfg.get("conf_thr"))

        if cfg['process'].get("region_adjacency_graph_l1") is not None:
            RAGl1_cfg = cfg['process']['region_adjacency_graph_l1']
            RAGl1_n_segments = int(RAGl1_cfg.get("n_segments"))

        if cfg['process'].get("remove_border_l1") is not None:
            RBl1_cfg = cfg['process']['remove_border_l1']
            RBl1_mask_ratio = float(RBl1_cfg.get("mask_ratio"))


        if cfg['process'].get("mask_otherRect_byArea") is not None:
            MOA_cfg = cfg['process']['mask_otherRect_byArea']
            MOA_MaskRect_ratio_thr = float(MOA_cfg.get("MaskRect_ratio_thr"))
            MOA_LocalgGobal_ratio_thr = float(MOA_cfg.get("LocalgGobal_ratio_thr"))
            MOA_area_ratio_thr = float(MOA_cfg.get("area_ratio_thr"))


        if cfg['process'].get("region_adjacency_graph_l2") is not None:
            RAGl2_cfg = cfg['process']['region_adjacency_graph_l2']
            RAGl2_n_segments = int(RAGl2_cfg.get("n_segments"))

        if cfg['process'].get("remove_border_l2") is not None:
            RBl2_cfg = cfg['process']['remove_border_l2']
            RBl2_mask_ratio = float(RBl2_cfg.get("mask_ratio"))


        if cfg['process'].get("mask_otherRect_byPoint") is not None:
            MOP_cfg = cfg['process']['mask_otherRect_byPoint']
            MOP_MaskRect_ratio_thr = float(MOP_cfg.get("MaskRect_ratio_thr"))
            MOP_LocalgGobal_ratio_thr = float(MOP_cfg.get("LocalgGobal_ratio_thr"))


        cleaner = cls(
            model_path=model_path,
            SWD_window_size=SWD_window_size,
            SWD_min_overlap=SWD_min_overlap,
            SWD_conf_thr=SWD_conf_thr,
            RAGl1_n_segments=RAGl1_n_segments,
            RBl1_mask_ratio=RBl1_mask_ratio,
            MOA_MaskRect_ratio_thr=MOA_MaskRect_ratio_thr,
            MOA_LocalgGobal_ratio_thr=MOA_LocalgGobal_ratio_thr,
            MOA_area_ratio_thr=MOA_area_ratio_thr,
            RAGl2_n_segments=RAGl2_n_segments,
            RBl2_mask_ratio=RBl2_mask_ratio,
            MOP_MaskRect_ratio_thr=MOP_MaskRect_ratio_thr,
            MOP_LocalgGobal_ratio_thr=MOP_LocalgGobal_ratio_thr,
            device=device,
        )
        return cleaner




















