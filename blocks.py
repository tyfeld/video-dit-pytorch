import torch
import torch.nn as nn 
import math
import torch.nn.functional as F
class PositionEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 
        self.inv_freq = 1. / (10000**(torch.arange(0, dim//2, 2).float() /(dim//2)))

        # inv_freq shape: dim // 4

    def get_1d_sincos_pos_embed(self, grid):
        out = torch.einsum("i, j -> ij", grid, self.inv_freq)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=-1)
        
        


    def forward(self, H, W, device):
        self.inv_freq = self.inv_freq.to(device)
        range_h = torch.arange(H, device=device) * 32 / H
        range_w = torch.arange(W, device=device) * 32 / W

        # scale h, w -> 32
        # 512p -> vae -> 64p -> patchify -> 32

        # grid_w = [0, 0, 0, 1, 1, 1, ... h ,h ,h ]
        # grid_h = [0, 1, 2, ...]

        # torch.meshgrid
        grid_h, grid_w = torch.meshgrid(range_w, range_h, indexing="ij")
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self.get_1d_sincos_pos_embed(grid_h)
        emb_w = self.get_1d_sincos_pos_embed(grid_w)

        return torch.cat([emb_h, emb_w], dim=-1).unsqueeze(0)
    
class TimeEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(256, 1152),
            nn.SiLU(),
            nn.Linear(1152, 1152)
        )

    def timestep_embedding(self, t, dim=256): 
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half).to(t.device)
        out = t[:, None] * freqs[None]
        embeddings = torch.cat([torch.cos(out), torch.sin(out)], dim=-1)
        return embeddings
    
    def forward(self, timestep):
        return self.ffn(self.timestep_embedding(timestep, 256))

class FpsEmbedder(TimeEmbedder):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(256, 1152),
            nn.SiLU(),
            nn.Linear(1152, 1152)
        )
    
    def forward(self, fps, b):
        # timestep: [1000, 1000], fps: [28]
        fps = fps.unsqueeze(0).repeat(b, 1).reshape(-1)
        # [28] -> [28, 28]
        return self.ffn(self.timestep_embedding(fps, 256)) 


class TextEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(4096, 1152),
            nn.GELU(approximate="tanh"),
            nn.Linear(1152, 1152)
        )

    def forward(self, context):
        return self.ffn(context)

class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=4, out_channels=1152, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        # padding 
        _, _, _, H, W = x.shape
        if W % 2 != 0 :
            x = F.pad(x, (0, 1))
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        return self.conv(x)


