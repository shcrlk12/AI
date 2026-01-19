import os
import random
import numpy as np
import multiprocessing as mp

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision import transforms
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from PIL import Image
import matplotlib.pyplot as plt


# -------------------------
# 재현성(선택)
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -------------------------
# COCO person binary seg Dataset
# -------------------------
class CocoPersonBinarySeg(CocoDetection):
    """
    입력: COCO images + instances json
    출력: (image_tensor [3,H,W], mask_tensor [1,H,W])  (person=1, background=0)
    """
    def __init__(self, img_root, ann_file, resize=(256, 256), train=True):
        super().__init__(img_root, ann_file)
        self.resize = resize
        self.train = train

        # 'person'의 category_id를 안전하게 얻기
        person_ids = self.coco.getCatIds(catNms=["person"])
        if len(person_ids) == 0:
            raise RuntimeError("COCO annotations에서 'person' category를 찾지 못했습니다.")
        self.person_cat_id = person_ids[0]

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)   # img: PIL, anns: list[dict]
        # img는 PIL RGB
        img = img.convert("RGB")

        # person 마스크 합치기 (H,W) uint8
        # annToMask는 (H,W) 0/1 반환
        H, W = img.size[1], img.size[0]
        mask = np.zeros((H, W), dtype=np.uint8)

        for a in anns:
            if a.get("category_id") != self.person_cat_id:
                continue
            m = self.coco.annToMask(a)  # (H,W) {0,1}
            if m is None:
                continue
            mask = np.maximum(mask, m.astype(np.uint8))

        # PIL/넘파이 -> 텐서 변환 + (이미지/마스크) 동일 변환
        # 1) resize (둘 다)
        img = F.resize(img, self.resize, interpolation=F.InterpolationMode.BILINEAR)
        mask_pil = Image.fromarray(mask * 255)  # 보기 좋게 0/255로
        mask_pil = F.resize(mask_pil, self.resize, interpolation=F.InterpolationMode.NEAREST)

        # 2) (train일 때만) 랜덤 좌우반전 - 이미지/마스크 같이
        if self.train and random.random() < 0.5:
            img = F.hflip(img)
            mask_pil = F.hflip(mask_pil)

        # 3) Tensor
        img_t = F.to_tensor(img)  # [3,H,W], 0~1
        mask_t = torch.from_numpy((np.array(mask_pil) > 0).astype(np.float32))  # [H,W] 0/1 float
        mask_t = mask_t.unsqueeze(0)  # [1,H,W]

        return img_t, mask_t


# -------------------------
# 간단 학습/평가
# -------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)  # [B,1,H,W] float(0/1)

        out = model(X)["out"]  # [B,1,H,W] (우리는 binary로 만들 것)
        loss = nn.functional.binary_cross_entropy_with_logits(out, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_iou(model, loader, device, thresh=0.5, show_n=3):
    model.eval()
    ious = []
    shown = 0

    for X, y in loader:
        X = X.to(device)   # (B,3,H,W)
        y = y.to(device)   # (B,1,H,W)

        logits = model(X)["out"]              # (B,1,H,W)
        pred = (torch.sigmoid(logits) > thresh).float()

        # IoU
        inter = (pred * y).sum(dim=(1,2,3))
        union = ((pred + y) > 0).float().sum(dim=(1,2,3))
        iou = (inter / (union + 1e-6)).detach().cpu().numpy()
        ious.extend(iou.tolist())

    return float(np.mean(ious)) if len(ious) else 0.0


@torch.no_grad()
def show_prediction(model, dataset, device, idx=0):
    model.eval()
    X, y = dataset[idx]
    logits = model(X.unsqueeze(0).to(device))["out"][0,0].cpu()
    pred = (torch.sigmoid(logits) > 0.5).float()

    img = X.permute(1,2,0).numpy()
    gt = y[0].numpy()

    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1); plt.imshow(img); plt.title("image"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(gt, cmap="gray"); plt.title("gt(person)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(pred.numpy(), cmap="gray"); plt.title("pred"); plt.axis("off")
    plt.tight_layout()
    plt.show()

import math

@torch.no_grad()
def save_all_after_epoch(
    model,
    loader,
    device,
    epoch,
    out_dir="vis",
    thresh=0.5,
    per_page=8,   # 한 페이지에 몇 장(샘플) 저장할지
    ncols=4       # 한 페이지에서 가로로 몇 장 배치할지
):
    """
    loader 전체를 돌며 image/gt/pred를 per_page개씩 묶어
    out_dir/epoch_XX_page_YYY.png 로 저장한다.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    total_cols = 3 * ncols
    buf = []
    page_idx = 1

    def flush_page(buf, page_idx):
        if not buf:
            return
        n = len(buf)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=total_cols, figsize=(3*total_cols, 3*nrows))
        if nrows == 1:
            axes = np.expand_dims(axes, axis=0)

        # 축 모두 끄기
        for r in range(nrows):
            for c in range(total_cols):
                axes[r, c].axis("off")

        for i, (img, gt, pr, iou_val) in enumerate(buf):
            r = i // ncols
            c0 = (i % ncols) * 3

            axes[r, c0 + 0].imshow(img)
            axes[r, c0 + 0].set_title("image")
            axes[r, c0 + 0].axis("off")

            axes[r, c0 + 1].imshow(gt, cmap="gray")
            axes[r, c0 + 1].set_title("gt")
            axes[r, c0 + 1].axis("off")

            axes[r, c0 + 2].imshow(pr, cmap="gray")
            axes[r, c0 + 2].set_title(f"pred IoU={iou_val:.3f}")
            axes[r, c0 + 2].axis("off")

        fig.suptitle(f"Epoch {epoch} | page {page_idx}", fontsize=16)
        plt.tight_layout()

        save_path = os.path.join(out_dir, f"epoch_{epoch:02d}_page_{page_idx:03d}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    for X, y in loader:
        X = X.to(device)   # (B,3,H,W)
        y = y.to(device)   # (B,1,H,W)

        logits = model(X)["out"]                 # (B,1,H,W)
        pred = (torch.sigmoid(logits) > thresh).float()

        inter = (pred * y).sum(dim=(1,2,3))
        union = ((pred + y) > 0).float().sum(dim=(1,2,3))
        iou = (inter / (union + 1e-6)).detach().cpu().numpy()

        B = X.size(0)
        for b in range(B):
            img = X[b].detach().cpu().permute(1,2,0).numpy()
            gt  = y[b,0].detach().cpu().numpy()
            pr  = pred[b,0].detach().cpu().numpy()
            buf.append((img, gt, pr, float(iou[b])))

            if len(buf) >= per_page:
                flush_page(buf, page_idx)
                buf = []
                page_idx += 1

    # 남은 것 저장
    flush_page(buf, page_idx)

def main():
    seed_everything(42)

    # CPU만 쓴다고 했으니
    device = "cpu"
    print("Using", device)

    # ===== 경로 수정 =====
    COCO_ROOT = r"C:\Users\Owner\Downloads\archive\coco2017"
    train_img = os.path.join(COCO_ROOT, "train2017")
    ann_dir = os.path.join(COCO_ROOT, "annotations")
    train_ann = os.path.join(ann_dir, "instances_train2017.json")

    # Dataset
    ds = CocoPersonBinarySeg(train_img, train_ann, resize=(256,256), train=True)

    # person category id
    person_id = ds.person_cat_id

    # person이 있는 이미지 인덱스만 모으기
    person_img_indices = []
    for i in range(len(ds)):
        img_id = ds.ids[i]  # COCO image_id
        ann_ids = ds.coco.getAnnIds(imgIds=[img_id], catIds=[person_id], iscrowd=None)
        if len(ann_ids) > 0:
            person_img_indices.append(i)

    print("person images:", len(person_img_indices))

    # 500장 랜덤 샘플
    n = 500
    random.shuffle(person_img_indices)
    picked = person_img_indices[:n]

    small = Subset(ds, picked)

    # DataLoader (윈도우/CPU 안전)
    loader = DataLoader(small, batch_size=2, shuffle=True, num_workers=0)

    # 모델: LRASPP MobileNetV3 (가벼운 세그)
    model = lraspp_mobilenet_v3_large(weights=None, num_classes=1)  # binary -> 1채널 logits
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 학습
    epochs = 30
    for ep in range(1, epochs+1):
        loss = train_one_epoch(model, loader, optimizer, device)
        iou = eval_iou(model, loader, device)
        print(f"epoch {ep:02d} | loss {loss:.4f} | IoU {iou:.3f}")

        # ✅ 에폭 끝날 때 전체 결과 저장
        save_all_after_epoch(
            model,
            loader,
            device,
            epoch=ep,
            out_dir="vis",
            thresh=0.5,
            per_page=8,
            ncols=4
        )
        print("saved pages to ./vis/")



    torch.save(model.state_dict(), "person_lraspp_cpu_50.pth")
    print("saved person_lraspp_cpu_50.pth")


if __name__ == "__main__":
    mp.freeze_support()
    main()
