import torch 
import torch.nn as nn 
from blocks import PositionEmbedder, PatchEncoder, TimeEmbedder, FpsEmbedder, TextEmbedder

class DiTBlock(nn.Module):
    def __init__(self):
        super().__init__()


class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedder = PositionEmbedder(1152)
        self.patch_encoder = PatchEncoder()
        self.time_embedder = TimeEmbedder()
        self.fps_embedder = FpsEmbedder()
        self.time_ffn = nn.Sequential(nn.SiLU(), nn.Linear(1152, 1152 * 6))
        self.text_embedder = TextEmbedder()

        # self.spatial_blocks = nn.ModuleList([DitBlock() for _ in range(28)])

        # self.temporal_blocks = nn.ModuleList([DitBlock() for _ in range(28)])


    def forward(self, x, timestep, fps, context, mask=None):
        # x is the input latent tensor, shape (Batch_Size, Channel, Frames, Height, Width)

        # Step1: Patchify and get position embeddings 
        B, C, Fx, Hx, Wx = x.shape 

        # Patch: 1, 2, 2
        patch_size = [1, 2, 2]
        F, H, W = Fx // patch_size[0], (Hx + 1) // patch_size[1], (Wx + 1) // patch_size[2]
        # 15, 23
        # patch numbers = 15*23 = 345
        pos_embed = self.pos_embedder(H, W, x.device)
        # 1, (HW), 1152

        # Step2: Timestep & FPS embedding
        # timestep: [1000, 1000], fps [FPS]
        timestep_embed = self.time_embedder(timestep)
        fps_embed = self.fps_embedder(fps, B)
        t = timestep_embed + fps_embed
        t = self.time_ffn(t)

        # Step3: text embedding ffn 
        # context: B, 1, token, 4096  -> B, 1, token, 1152
        y = self.text_embedder(context)
        # B, 1, token, 1152 -> B, 1, valid_token, 1152
        # mask: [1, token] -> [1, 300] 
        mask = mask.repeat(B, 1).unsqueeze(-1)   # B, token, 1
        y = y.squeeze(1) # B, token, 1152
        y = y.masked_select(mask).reshape(1, -1, 1152) # B, valid_token, 1152
        # [valid_token] * B
        valid_token_nums_list = [y.shape[1] // B] * B

        # Step4: emb x 
        x = self.patch_encoder(x)  # B, 1152, F, H, W
        # reshape -> B, F, (HW), 1152
        x = x.reshape(B, F, -1, 1152) 
        x = x + pos_embed

        print(x.shape)  # 2, 16, 345, 1152






if __name__ == "__main__":
    model = DiT()
    x = torch.randn(2,4,16,30,45)
    timestep = torch.tensor([1000, 1000])
    fps = torch.tensor([24])
    context = torch.randn(2, 1, 300, 4096)
    mask = torch.zeros(1, 300, dtype=torch.bool)
    mask[0, :12] = True
    # x, timestep, fps, context, mask
    y = model(x, timestep, fps, context, mask) # 1 * 345 * 1152
        # x : B * N * D
        # x + pos_embed
        





    