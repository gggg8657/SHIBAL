import torch
from torch import nn
from utils import FeedForward, DECOUPLED
from performer_pytorch import Performer

class AttnBlock(nn.Module):
    def __init__(self, dim, depth, dropout, attn_dropout, heads = 16, ff_mult = 2):
        super().__init__()
        self.performer = Performer(dim = dim, 
                                   depth = depth, 
                                   heads = heads, 
                                   dim_head = dim // heads, 
                                   causal = False,
                                   ff_mult = ff_mult,
                                   local_attn_heads = 8,
                                   local_window_size = dim // 8,
                                   ff_dropout = dropout,
                                   attn_dropout = attn_dropout,
                                   )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B, -1, C)
        x = self.performer(x)
        x = x.view(B, T, H, W, C)
        return x

class ConvBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        dropout = 0.,
        heads = 16,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.conv = DECOUPLED(dim, heads)
        self.ff = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# main class

class Model(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.2,
        attn_dropout = 0.1,
        ff_mult = 4,
        dims = (192, 128),
        depths = (3, 3),
        block_types = ('c', 'a')
    ):
        dims = dims
        depths = depths
        block_types = block_types
        super().__init__()
        self.init_dim, *_, last_dim = dims

        self.stages = nn.ModuleList([])

        for ind, (depth, block_types) in enumerate(zip(depths, block_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            
            if block_types == "c":
                for _ in range(depth):
                    self.stages.append(
                        ConvBlock(
                            dim = stage_dim,
                            ff_mult=ff_mult,
                            dropout = dropout,
                        )
                    )
            elif block_types == "a":
                for _ in range(depth):
                    self.stages.append(AttnBlock(stage_dim, 1, dropout, attn_dropout, ff_mult=ff_mult))
                
            if not is_last:
                self.stages.append(
                    nn.Sequential(
                        nn.LayerNorm(stage_dim),
                        nn.Linear(stage_dim, dims[ind + 1])
                    )
                )

        self.norm0 = nn.LayerNorm(192)
        self.linear = nn.Linear(192, dims[0])
        self.norm = nn.LayerNorm(last_dim)
        self.fc = nn.Linear(last_dim, 1)
        self.drop_out = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        
    def forward(self, x):
        # 입력 텐서 차원 정리
        if x.dim() == 6:
            # 흔히 (B, 1, C, T, H, W) 또는 (B, 1, T, C, H, W) 형태가 들어올 수 있음
            # 우선 크기가 1인 차원은 제거 (배치/채널 외 단일 차원)
            if x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                # 일반화: 모든 singleton 차원 제거 (배치 제외)
                b = x.shape[0]
                x = x.view(b, *[d for d in x.shape[1:] if d != 1])
        if x.dim() != 5:
            raise ValueError(f"예상되지 않은 텐서 차원: {x.dim()}, 형태: {x.shape}")

        # 채널 차원 자동 감지 후 (B, T, H, W, C)로 정렬
        # 후보 채널 크기: 초기 채널 self.init_dim (tiny=32, base=192)
        b, d1, d2, d3, d4 = x.shape
        # 가능한 입력 형태: (B, C, T, H, W) 또는 (B, T, C, H, W)
        if d1 == self.init_dim:
            # (B, C, T, H, W) -> (B, T, H, W, C)
            x = x.permute(0, 2, 3, 4, 1)
        elif d2 == self.init_dim:
            # (B, T, C, H, W) -> (B, T, H, W, C)
            x = x.permute(0, 1, 3, 4, 2)
        else:
            # base 모델일 때 라벨링된 피처가 (B, 192, 16, 10, 10) 형태일 가능성이 높음
            # 마지막 축이 채널이 되도록 가장 그럴듯한 축을 채널로 가정
            if d1 in (32, 64, 128, 192):
                x = x.permute(0, 2, 3, 4, 1)
            elif d2 in (32, 64, 128, 192):
                x = x.permute(0, 1, 3, 4, 2)
            else:
                raise ValueError(f"채널 차원을 감지할 수 없습니다: {x.shape}, init_dim={self.init_dim}")

        # 이제 x는 (B, T, H, W, C)
        if x.shape[-1] != self.init_dim:
            x = self.linear(self.norm0(x))

        for stage in self.stages:
            x = stage(x)

        # (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pooling(x).squeeze()

        x = self.drop_out(x)
        x = self.norm(x)
        logits = self.fc(x)
        return logits, x
