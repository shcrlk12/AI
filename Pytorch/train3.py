import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

# 학습 때와 동일하게!
RESIZE = (256, 256)

def load_model(weight_path, device="cpu"):
    model = lraspp_mobilenet_v3_large(weights=None, num_classes=1)
    sd = torch.load(weight_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict_person_mask(model, img_path, device="cpu", thresh=0.5):
    img = Image.open(img_path).convert("RGB")

    # 전처리: 학습과 동일한 resize + to_tensor
    img_r = F.resize(img, RESIZE, interpolation=F.InterpolationMode.BILINEAR)
    x = F.to_tensor(img_r).unsqueeze(0).to(device)  # [1,3,H,W]

    logits = model(x)["out"][0, 0]          # [H,W]
    prob = torch.sigmoid(logits).cpu().numpy()
    mask = (prob > thresh).astype(np.uint8) # 0/1

    return img_r, prob, mask

def overlay_mask(img_pil, mask01, alpha=0.5):
    """
    img_pil: PIL RGB (RESIZE된 이미지)
    mask01: (H,W) 0/1
    """
    img = np.array(img_pil).astype(np.float32) / 255.0
    mask = mask01.astype(bool)

    # 빨간색으로 칠하기(간단)
    overlay = img.copy()
    overlay[mask, 0] = 1.0  # R 채널 올림
    overlay[mask, 1] *= 0.3
    overlay[mask, 2] *= 0.3

    out = (1 - alpha) * img + alpha * overlay
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return out

def run_infer():
    device = "cpu"
    weight_path = "person_lraspp_cpu_500.pth"
    # img_path = r"C:\Users\Owner\Downloads\archive\coco2017\val2017\000000578871.jpg"  # 아무 이미지로 바꾸세요
    img_path = r"C:\Users\Owner\Downloads\archive\coco2017\val2017\000000581357.jpg"  # 아무 이미지로 바꾸세요

    model = load_model(weight_path, device=device)
    img_r, prob, mask = predict_person_mask(model, img_path, device=device, thresh=0.5)

    over = overlay_mask(img_r, mask, alpha=0.55)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_r); plt.title("input(resized)"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(prob, cmap="gray"); plt.title("prob map"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(over); plt.title("overlay"); plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_infer()
