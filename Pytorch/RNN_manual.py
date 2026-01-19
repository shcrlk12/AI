import torch
import torch.nn as nn

class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # W_xh, W_hh, W_hy를 Linear로 표현(편하게)
        self.x_to_h = nn.Linear(input_size, hidden_size)      # W_xh, b 포함
        self.h_to_h = nn.Linear(hidden_size, hidden_size, bias=False)  # W_hh (bias는 위에 있으니 생략 가능)
        self.h_to_y = nn.Linear(hidden_size, output_size)     # W_hy, b

    def forward(self, x, h0=None):
        """
        x: (B, T, input_size)
        h0: (B, hidden_size) 또는 None
        """
        B, T, _ = x.shape
        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h = h0

        outputs = []
        for t in range(T):
            xt = x[:, t, :]                      # (B, input_size)
            h = torch.tanh(self.x_to_h(xt) + self.h_to_h(h))  # (B, hidden_size)
            yt = self.h_to_y(h)                  # (B, output_size)
            outputs.append(yt)

        y = torch.stack(outputs, dim=1)          # (B, T, output_size)
        return y, h                               # 전체 출력, 마지막 hidden

# ---- 사용 예시 ----
B, T, Fin = 4, 10, 3
Fout = 2
x = torch.randn(B, T, Fin)

model = ManualRNN(input_size=Fin, hidden_size=32, output_size=Fout)
y, h_last = model(x)

print("y:", y.shape)         # (4,10,2)
print("h_last:", h_last.shape)  # (4,32)
