import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.layer1 = vgg[:2]   
        self.layer2 = vgg[2:7]  
        self.layer3 = vgg[7:12] 
        self.layer4 = vgg[12:21] 
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

def decompose_image(img, blur_radius=15):
    """Decomposes image into Base layer (low frequency) and Detail layer (high frequency)"""
    base = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)
    detail = img - base
    return base, detail

def l1_norm_spatial_frequency(feat_maps):
    """Calculates the L1-norm over the channel dimension to yield a spatial activity map"""
    l1_norm = torch.sum(torch.abs(feat_maps), dim=1).squeeze(0)
    return l1_norm.cpu().numpy()

def fuse_detail_layers(detail_vis, detail_ir, model, device):
    """Fuses detail layers using multi-layer VGG19 features + L1 Norm Max Selection"""
    # Normalize inputs to [0, 1] range expected by VGG19
    t_vis = torch.from_numpy(np.repeat(detail_vis[None, None, ...], 3, axis=1)).float().to(device) / 255.0
    t_ir = torch.from_numpy(np.repeat(detail_ir[None, None, ...], 3, axis=1)).float().to(device) / 255.0
    
    with torch.no_grad():
        feats_vis = model(t_vis)
        feats_ir = model(t_ir)
        
    kernel = np.ones((3, 3), np.float32) / 9.0
    
    weight_vis_total = np.zeros_like(detail_vis, dtype=np.float32)
    weight_ir_total = np.zeros_like(detail_ir, dtype=np.float32)
    
    for f_vis, f_ir in zip(feats_vis, feats_ir):
        l1_vis = l1_norm_spatial_frequency(f_vis)
        l1_ir = l1_norm_spatial_frequency(f_ir)
        
        l1_vis_res = cv2.resize(l1_vis, (detail_vis.shape[1], detail_vis.shape[0]), interpolation=cv2.INTER_CUBIC)
        l1_ir_res = cv2.resize(l1_ir, (detail_ir.shape[1], detail_ir.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        map_vis = cv2.filter2D(l1_vis_res, -1, kernel)
        map_ir = cv2.filter2D(l1_ir_res, -1, kernel)
        
        weight_vis_total += map_vis
        weight_ir_total += map_ir

    # Max selection strategy chooses the most prominent structural feature
    mask_vis = (weight_vis_total >= weight_ir_total).astype(np.float32)
    mask_ir = 1.0 - mask_vis
    
    fused_detail = (mask_vis * detail_vis) + (mask_ir * detail_ir)
    return fused_detail

def fuse_images(vis_path, ir_path, model, device):
    # Load BOTH images as Grayscale to match Li et al.'s exact framework
    img_vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE) 
    img_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE) 
    
    if img_vis is None or img_ir is None:
        raise FileNotFoundError("One or both input images could not be loaded. Check paths.")

    if img_vis.shape != img_ir.shape:
        img_ir = cv2.resize(img_ir, (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Step 1: Base and Detail Decomposition
    base_vis, detail_vis = decompose_image(img_vis.astype(np.float32))
    base_ir, detail_ir = decompose_image(img_ir.astype(np.float32))
    
    # Step 2: Base Fusion (Weight averaging)
    fused_base = (base_vis * 0.5) + (base_ir * 0.5)
    
    # Step 3: Detail Fusion via VGG19
    fused_detail = fuse_detail_layers(detail_vis, detail_ir, model, device)
    
    # Step 4: Reconstruct Final High-Contrast Grayscale Image
    final_fused = np.clip(fused_base + fused_detail, 0, 255).astype(np.uint8)
    
    return final_fused

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running fusion framework on device: {device}")
    
    extractor = VGG19FeatureExtractor().to(device)
    
    # Explicit Grid5000 Cluster Directory Targets
    visible_img_path = "/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/audio_visual/corsican/all_db_fire/seq05_rgb_013.png" 
    infrared_img_path = "/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/audio_visual/corsican/all_db_fire/seq05_nir_013.png"
    output_path = "corsican_vgg19_fused_output.png"
    
    try:
        fused_result = fuse_images(visible_img_path, infrared_img_path, extractor, device)
        cv2.imwrite(output_path, fused_result)
        print(f"Successfully generated and saved grayscale fused image to: {output_path}")
    except Exception as e:
        print(f"Execution Error: {e}")