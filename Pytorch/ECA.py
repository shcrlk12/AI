import torch
import torch.nn as nn

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA)
    입력:  x (N, C, H, W)
    출력:  x * a (N, C, H, W)  (채널별로 가중치 곱한 결과)
    """
    def __init__(self, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1, "k_size는 보통 홀수로 씁니다."
        self.gap = nn.AdaptiveAvgPool2d(1)  # (N,C,H,W) -> (N,C,1,1)
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=k_size, padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape

        # 1) 채널별 요약(GAP): (N,C,1,1)
        y = self.gap(x)

        # 2) 1D conv를 "채널축"으로 적용하기 위해 모양 변환
        #    (N,C,1,1) -> (N,1,C)
        y = y.squeeze(-1).transpose(1, 2)   # (N,1,C)

        # 3) 채널 벡터에 1D conv: (N,1,C)
        y = self.conv1d(y)

        # 4) 다시 (N,C,1,1)로 복원 후 sigmoid
        y = y.transpose(1, 2).unsqueeze(-1)  # (N,C,1,1)
        a = self.sigmoid(y)                  # 채널 가중치

        # 5) 원래 x에 채널별로 곱하기 (broadcast됨)
        return x * a
