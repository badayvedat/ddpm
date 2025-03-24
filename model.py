import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10_000.0)) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.emb = nn.Parameter(self.emb, requires_grad=False)
    
    def forward(self, t: torch.Tensor):
        pos = t.float().unsqueeze(1)    # [N, 1]
        emb = pos * self.emb.unsqueeze(0)   # [N, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)    # [N, embedding_dim]
        return emb


class TimeDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, time_emb_dim: int = 128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        self.enc1 = TimeDoubleConv(3, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = TimeDoubleConv(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = TimeDoubleConv(128, 256, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = TimeDoubleConv(256, 512, time_emb_dim)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = TimeDoubleConv(512, 256, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = TimeDoubleConv(256, 128, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = TimeDoubleConv(128, 64, time_emb_dim)

        self.out = nn.Conv2d(64, 3, 1)

    
    def forward(self, x, t):
        t_emb = self.time_emb(t).to(x.dtype)

        e1 = self.enc1(x, t_emb)      # (batch_size, 64, 32, 32)
        p1 = self.pool1(e1)           # (batch_size, 64, 16, 16)
        e2 = self.enc2(p1, t_emb)     # (batch_size, 128, 16, 16)
        p2 = self.pool2(e2)           # (batch_size, 128, 8, 8)
        e3 = self.enc3(p2, t_emb)     # (batch_size, 256, 8, 8)
        p3 = self.pool3(e3)           # (batch_size, 256, 4, 4)

        b = self.bottleneck(p3, t_emb)  # (batch_size, 512, 4, 4)

        u3 = self.up3(b)              # (batch_size, 256, 8, 8)
        cat3 = torch.cat([u3, e3], dim=1)  # (batch_size, 512, 8, 8)
        d3 = self.dec3(cat3, t_emb)   # (batch_size, 256, 8, 8)
        u2 = self.up2(d3)             # (batch_size, 128, 16, 16)
        cat2 = torch.cat([u2, e2], dim=1)  # (batch_size, 256, 16, 16)
        d2 = self.dec2(cat2, t_emb)   # (batch_size, 128, 16, 16)
        u1 = self.up1(d2)             # (batch_size, 64, 32, 32)
        cat1 = torch.cat([u1, e1], dim=1)  # (batch_size, 128, 32, 32)
        d1 = self.dec1(cat1, t_emb)   # (batch_size, 64, 32, 32)

        out = self.out(d1)            # (batch_size, 3, 32, 32)
        return out
