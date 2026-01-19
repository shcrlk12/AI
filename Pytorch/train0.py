import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------
# 재현성(선택)
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # 속도/성능 이점(재현성은 약간 흔들릴 수 있음)

seed_everything(42)

# -----------------------
# Device
# -----------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using", device)

# -----------------------
# Transform (증강 + 정규화)
# FashionMNIST mean/std (자주 쓰는 값)
# -----------------------
mean, std = 0.2860, 0.3530

train_tf = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# -----------------------
# Dataset / DataLoader
# -----------------------
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=train_tf
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=test_tf
)


batch_size = 128
train_loader = DataLoader(training_data, batch_size=128, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False, num_workers=0)

from torchvision.utils import make_grid

X, y = next(iter(train_loader))  # (B,1,28,28)
grid = make_grid(X[:32], nrow=8, padding=2)  # 32장


# -----------------------
# Model (더 안정적인 CNN + GAP)
# -----------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(1, 32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(2),           # 28 -> 14
            nn.Dropout(0.10),

            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2),           # 14 -> 7
            nn.Dropout(0.15),

            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 7x7 -> 1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        print(x)
        fm = x[0, 0].detach().cpu().numpy()  # (H,W)
        plt.imshow(fm, cmap="gray")
        plt.title(f"feature map: step (ch=0)")
        plt.axis("off")
        plt.show()

        x = self.features(x)

        fm = x[0, 0].detach().cpu().numpy()  # (H,W)
        plt.imshow(fm, cmap="gray")
        plt.title(f"feature map: step (ch=0)")
        plt.axis("off")
        plt.show()

        fm = x[0, 1].detach().cpu().numpy()  # (H,W)
        plt.imshow(fm, cmap="gray")
        plt.title(f"feature map: step (ch=0)")
        plt.axis("off")
        plt.show()

        x = self.pool(x)
        x = self.classifier(x)
        return x

model = Net().to(device)
print(model)

# -----------------------
# Loss / Optimizer / Scheduler
# -----------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)

# OneCycleLR: FashionMNIST에서 꽤 잘 먹습니다.
epochs = 30
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.2,
    div_factor=10.0,
    final_div_factor=100.0
)

use_amp = (device == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# -----------------------
# Train / Test
# -----------------------
def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(X)
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        total_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total

best_acc = 0.0
for epoch in range(1, epochs + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader)
    te_loss, te_acc = evaluate(model, test_loader)

    if te_acc > best_acc:
        best_acc = te_acc
        torch.save(model.state_dict(), "best_fmnist_cnn.pt")

    print(f"Epoch {epoch:02d} | "
          f"train: loss {tr_loss:.4f}, acc {tr_acc*100:.2f}% | "
          f"test:  loss {te_loss:.4f}, acc {te_acc*100:.2f}% | "
          f"best {best_acc*100:.2f}%")

print("Done!")
print("Saved: best_fmnist_cnn.pt (state_dict)")

# 전체 모델로 저장하고 싶으면(권장 X: 환경 의존)
# torch.save(model, "my_full_model.pt")
