import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module): 
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__() 
        
        assert img_size % patch_size == 0, 'input image dimensions must be divisible by the patch size'
        
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_dim = patch_size ** 2 * in_channels 
        
        self.embedding = nn.Sequential(
                               Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size), 
                               nn.Linear(self.patch_dim, embed_dim))
        
        # 4 x 16 x 3 x 224 x 224 -> 4 x 16 x 196 x 768 -> 4 x 16 x 196 x 192 
        
    def forward(self, x): 
        
        """
        Args:
            x: input video frame (4, 16, 3, 224, 224)
        """
        
        x = self.embedding(x)
        return x

class MultiHeadAttention(nn.Module): 
    
    def __init__(self, embed_dim = 768, n_heads = 12, drop=0.1): 
        
        super().__init__()
        
        assert embed_dim % n_heads == 0, 'the dimension of input embedding vector must be divisble by the number of attention head'
        
        self.embed_dim = embed_dim 
        self.head_num = n_heads 
        self.scale = (embed_dim // n_heads) ** -0.5 
        
        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=True)
        
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x): 
        
        """
        Args: 
            x: the input vector in [batch_size, num_frames, seq_length, dim] or
                                   [batch_size, num_frames, dim]
        """
        if x.dim() == 4: 
            B, T, N, D = x.shape 
            qkv = self.qkv(x).reshape(B, T, 3, N, self.head_num, D // self.head_num).permute(2, 0, 1, 4, 3, 5)
            # 4 x 16 x 197 x 768 -> 4 x 16 x 197 x (768*3) -> 4 x 16 x 3 x 197 x 12 x 64 -> 3 x 4 x 16 x 12 x 197 x 64
        elif x.dim() == 3:
            B, T, D = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.head_num, D // self.head_num).permute(2, 0, 3, 1, 4)
            # 4 x 17 x 768 -> 4 x 17 x (768*3) -> 4 x 17 x 3 x 12 x 64 -> 3 x 4 x 12 x 17 x 64 

        q, k, v = qkv[0], qkv[1], qkv[2]   # 4 x 16 x 12 x 197 x 64 or 4 x 12 x 17 x 64
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale   #  4 x 16 x 12 x 197 x 197 or 4 x 12 x 17 x 17 
        
        scores = F.softmax(attn, dim = -1)   # 4 x 16 x 12 x 197 x 197 or 4 x 12 x 17 x 17 
        
        if x.dim == 4:
            x = torch.matmul(scores, v).permute(0, 1, 3, 2, 4).reshape(B, T, N, -1)
            # 4 x 16 x 12 x 197 x 64 -> 4 x 16 x 197 x 12 x 64 -> 4 x 16 x 197 x 768
            
        elif x.dim == 3:
            x = torch.matmul(scores, v).permute(0, 2, 1, 3).reshape(B, T, -1)
            # 4 x 12 x 17 x 64 -> 4 x 17 x 12 x 64
        
        x = self.proj(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    
    def __init__(self,embed_dim = 768, n_heads = 12, expansion_factor = 4, drop=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim 
        self.head_num = n_heads
        self.expansion_factor = expansion_factor
        
        self.attention = MultiHeadAttention(embed_dim, n_heads, drop)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor), 
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1))
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        
    def forward(self, x): 
        """
        Args: 
            x: input vector [4 x 16 x 197 x 762]
        """
        
        x = self.attention(self.norm1(x)) + x 
        
        x = self.mlp(self.norm2(x)) + x
        
        return x

class Transformer(nn.Module): 
    
    def __init__(self, embed_dim=192, n_heads=12, expansion_factor=4, L=4, drop=0.1): 
        super().__init__()
        
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, n_heads, expansion_factor, drop) for _ in range(L)])
        
    def forward(self, x):
        """
        Args: 
            x: input video frames 
            4 x 16 x 197 x 192
        """
     
        for layer in self.layers: 
            x = layer(x)
            
        return x

class ViViT(nn.Module): 
    
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_frames=300,
                 embed_dim=192, query_dim=64, n_heads=12, expansion_factor=4, L=4, drop=0.1):
        
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size'
        
        num_patches = (image_size // patch_size) ** 2
        
        
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim) 
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, embed_dim))
        
        
        self.space_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.space_transformer = Transformer(embed_dim, n_heads, expansion_factor, L, drop)
        
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.temporal_transformer = Transformer(embed_dim, n_heads, expansion_factor, L, drop) 
        

        self.mlp_dim = embed_dim + query_dim 

        self.mlp = nn.Sequential(
                            nn.Linear(self.mlp_dim, self.mlp_dim),
                            nn.ReLU(), 
                            nn.Linear(self.mlp_dim, self.mlp_dim), 
                            nn.ReLU(), 
                            nn.Linear(self.mlp_dim, 1))
        
        
    def forward(self, x, y): 
        
        """
        Args: 
            x: input video frames [batch_size, num_frames, channel_size, width, height]
            y: high dimension vector of the query vector (x, y, z, t)  [Batch_size, dimesion]  (4 x 64)
        """
        
        B, N, C, H, W = x.shape
        
        
        x = self.patch_embedding(x)                  # 4 x 16 x 3 x 224 x 224 -> 4 x 16 x 196 x 192
        
        space_cls_token = self.space_cls_token.repeat((B, N, 1, 1))    # 1 x 1 x 192 -> 4 x 16 x 1 x 192
        
        x = torch.cat((space_cls_token, x), dim=2)     # 4 x 16 x 197 x 192 
        
        x += self.pos_embedding[:, :N, :, :] 
        
        x = self.space_transformer(x)    # 4 x 16 x 197 x 192 
        
        x = x[:, :, 0, :]                # 4 x 16 x 192 
        
        temporal_cls_token = self.temporal_cls_token.repeat((B, 1, 1)) # 1 x 1 x 192 -> 4 x 1 x 192 
        
        x = torch.cat((temporal_cls_token, x), dim=1)
        
        x = self.temporal_transformer(x)

        
        x = torch.cat((x[:, 0, :], y.float()), dim=-1)
        
        return self.mlp(x)