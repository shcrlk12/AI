import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# 1) Manual RNN (nn.RNN 안 씀)
# -------------------------
class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x_to_h = nn.Linear(input_size, hidden_size)                 # W_xh + b
        self.h_to_h = nn.Linear(hidden_size, hidden_size, bias=False)    # W_hh
        self.h_to_y = nn.Linear(hidden_size, output_size)                # W_hy + b

    def forward(self, x, h0=None):
        """
        x: (B, T, input_size)
        """
        B, T, _ = x.shape
        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h = h0

        outputs = []
        for t in range(T):
            xt = x[:, t, :]  # (B, input_size)
            h = torch.tanh(self.x_to_h(xt) + self.h_to_h(h))
            yt = self.h_to_y(h)  # (B, output_size)
            outputs.append(yt)

        y = torch.stack(outputs, dim=1)  # (B, T, output_size)
        return y, h


# -------------------------
# 2) 유의미한 데이터: 사인파 다음값 예측
# -------------------------
class SineNextValueDataset(Dataset):
    def __init__(self, n_samples=2000, T=50, noise_std=0.05):
        self.T = T
        self.n = n_samples
        self.noise_std = noise_std

        # 각 샘플마다 (진폭, 주파수, 위상) 랜덤
        self.params = []
        for _ in range(n_samples):
            A = random.uniform(0.5, 1.5)
            f = random.uniform(0.02, 0.08)  # 너무 빠르지 않게
            phi = random.uniform(0, 2 * math.pi)
            self.params.append((A, f, phi))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        A, f, phi = self.params[idx]
        # t=0..T 까지 만들어서, 입력은 0..T-1, 정답은 T
        t = np.arange(self.T + 1, dtype=np.float32)
        y = A * np.sin(2 * math.pi * f * t + phi)

        if self.noise_std > 0:
            y = y + np.random.normal(0, self.noise_std, size=y.shape).astype(np.float32)

        x_seq = y[:self.T]          # (T,)
        y_next = y[self.T]          # 스칼라

        x_seq = torch.tensor(x_seq).unsqueeze(-1)     # (T,1)
        y_next = torch.tensor([y_next])               # (1,)
        return x_seq, y_next


# -------------------------
# 3) 학습 루프
# -------------------------
def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for x, y_next in loader:
        x = x.to(device)           # (B,T,1)
        y_next = y_next.to(device) # (B,1)

        y_all, _ = model(x)        # (B,T,1)
        pred_next = y_all[:, -1, :]  # 마지막 시점 출력만 사용 (B,1)

        loss = loss_fn(pred_next, y_next)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    for x, y_next in loader:
        x = x.to(device)
        y_next = y_next.to(device)
        y_all, _ = model(x)
        pred_next = y_all[:, -1, :]
        total += loss_fn(pred_next, y_next).item()
    return total / max(1, len(loader))


@torch.no_grad()
def show_some_predictions(model, loader, device, k=5):
    model.eval()
    x, y_next = next(iter(loader))
    x = x.to(device)
    y_next = y_next.to(device)

    y_all, _ = model(x)
    pred_next = y_all[:, -1, :]

    x0 = x[:k].cpu().squeeze(-1).numpy()
    gt = y_next[:k].cpu().squeeze(-1).numpy()
    pr = pred_next[:k].cpu().squeeze(-1).numpy()

    for i in range(k):
        print(f"[sample {i}] last_input={x0[i,-1]:+.3f} | gt_next={gt[i]:+.3f} | pred_next={pr[i]:+.3f}")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cpu"

    T = 50
    train_ds = SineNextValueDataset(n_samples=3000, T=T, noise_std=0.05)
    test_ds  = SineNextValueDataset(n_samples=600,  T=T, noise_std=0.05)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)

    model = ManualRNN(input_size=1, hidden_size=64, output_size=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 21):
        tr = train_one_epoch(model, train_loader, opt, device)
        te = eval_loss(model, test_loader, device)
        print(f"epoch {epoch:02d} | train MSE {tr:.5f} | test MSE {te:.5f}")

        if epoch in (1, 5, 10, 20):
            show_some_predictions(model, test_loader, device, k=3)

if __name__ == "__main__":
    main()
