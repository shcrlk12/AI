import os
import time
import multiprocessing as mp

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Subset

from PIL import Image

# pycocotools는 CocoDetection 내부에서 coco 객체로 접근 가능
# (mask 만들 때 coco.annToMask 사용)

def collate_fn(batch):
    # detection 모델은 (list[img], list[target]) 형태를 기대
    return tuple(zip(*batch))

class CocoInstanceSegDataset(CocoDetection):
    """
    COCO instance segmentation용 Dataset.
    - img: Tensor [3,H,W]
    - target: dict(boxes, labels, masks, image_id, area, iscrowd)
    """
    def __init__(self, img_root, ann_file, transforms_=None):
        super().__init__(img_root, ann_file)
        self.transforms_ = transforms_

        # COCO category_id(띄엄띄엄 존재) -> contiguous label(1..K)로 매핑
        cat_ids = self.coco.getCatIds()
        cat_ids = sorted(cat_ids)
        self.cat_id_to_contiguous = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.num_classes = len(cat_ids) + 1  # + background(0)

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)   # img: PIL, anns: list[dict]
        image_id = self.ids[idx]

        # PIL -> Tensor + augmentation
        if self.transforms_ is not None:
            img = self.transforms_(img)        # Tensor [3,H,W]
        else:
            img = transforms.ToTensor()(img)

        # 타겟 생성
        boxes = []
        labels = []
        masks = []
        area = []
        iscrowd = []

        # anns는 한 이미지의 annotation 리스트(객체들)
        for a in anns:
            # bbox: [x, y, w, h]
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue

            # category_id -> contiguous label
            cat_id = a["category_id"]
            if cat_id not in self.cat_id_to_contiguous:
                continue
            label = self.cat_id_to_contiguous[cat_id]

            # mask 만들기 (H,W) uint8 -> bool
            m = self.coco.annToMask(a)
            if m.sum() == 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(label)
            masks.append(m)
            area.append(a.get("area", float(w * h)))
            iscrowd.append(a.get("iscrowd", 0))

        if len(boxes) == 0:
            # detection 모델은 빈 타겟도 처리 가능하지만, 학습 안정성 위해 더미 한 개를 넣는 방식도 있음
            # 여기서는 "빈 타겟"을 허용 (이미지만 반환하면 모델이 에러낼 수 있어서 dict는 유지)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(masks, dtype=torch.uint8)  # [N,H,W]
            area = torch.tensor(area, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        return img, target


def get_model_instance_segmentation(num_classes):
    # COCO pretrained로 시작
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # box head 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # mask head 교체
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)   # dict of losses
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        # if i % print_freq == 0:
        loss_items = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
        print(f"[epoch {epoch} iter {i}/{len(data_loader)}] total={losses.item():.4f} detail={loss_items}")

    dt = time.time() - t0
    return running_loss / max(1, len(data_loader)), dt


@torch.no_grad()
def inference_one_image(model, device, img_path, score_thresh=0.5):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    x = transforms.ToTensor()(img).to(device)
    out = model([x])[0]

    keep = out["scores"] >= score_thresh
    result = {
        "boxes": out["boxes"][keep].cpu(),
        "labels": out["labels"][keep].cpu(),
        "scores": out["scores"][keep].cpu(),
        "masks": out["masks"][keep].cpu(),   # [N,1,H,W]
    }
    return result


def main():
    # ====== 경로 설정(본인 환경에 맞게 수정) ======
    COCO_ROOT = r"C:\Users\Owner\Downloads\archive\coco2017"  # 예: D:\coco2017
    train_img = os.path.join(COCO_ROOT, "train2017")
    val_img = os.path.join(COCO_ROOT, "val2017")
    ann_dir = os.path.join(COCO_ROOT, "annotations")
    train_ann = os.path.join(ann_dir, "instances_train2017.json")
    val_ann = os.path.join(ann_dir, "instances_val2017.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ====== Transform ======
    # detection용: ToTensor는 필수. 증강은 필요시 추가 가능.
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    # ====== Dataset ======
    train_ds = CocoInstanceSegDataset(train_img, train_ann, transforms_=train_tf)
    val_ds = CocoInstanceSegDataset(val_img, val_ann, transforms_=val_tf)

    train_ds_small = Subset(train_ds, range(50))  # 2천장만

    num_classes = train_ds.num_classes
    print("num_classes (incl background):", num_classes)

    # ====== DataLoader ======
    # 윈도우 안정성: num_workers=0 권장 (돌아가면 2~4로 올려도 됨)
    train_loader = DataLoader(
        train_ds_small, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # ====== Model ======
    model = get_model_instance_segmentation(num_classes).to(device)

    # ====== Optimizer / Scheduler ======
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # ====== Train ======
    epochs = 5
    for epoch in range(1, epochs + 1):
        avg_loss, sec = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        lr_scheduler.step()
        print(f"Epoch {epoch} done. avg_loss={avg_loss:.4f}, time={sec:.1f}s")

        # 체크포인트 저장
        torch.save(model.state_dict(), f"maskrcnn_coco_epoch{epoch}.pth")

    print("Training finished.")

    # ====== Inference demo ======
    # val 이미지 하나 골라 보기
    sample_img = os.path.join(val_img, os.listdir(val_img)[0])
    pred = inference_one_image(model, device, sample_img, score_thresh=0.7)
    print("Inference:", {k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in pred.items()})
    print("Saved weights:", f"maskrcnn_coco_epoch{epochs}.pth")


if __name__ == "__main__":
    mp.freeze_support()  # 윈도우 필수 습관
    main()
