import torch
import torch.nn as nn
from transformers import MobileViTConfig, MobileViTModel
from .common import SwiGLUMLP, init_linear_orthogonal
import numpy as np


class VisionEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, in_channels=3, img_size=224):
        super().__init__()
        self.latent_dim = latent_dim
        
        # A tiny, randomly initialized CNN backbone
        # Takes (B, in_channels, H, W) -> outputs a spatial feature map
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1), # 224x224 -> 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # 112x112 -> 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # 56x56 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))                                    # Global pool to 1x1 spatial size
        )
        
        # The output channels from our last conv layer (equivalent to your old in_features)
        cnn_out_features = 64 
        
        self.projection = SwiGLUMLP(
            input_dim=cnn_out_features,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            norm_fn=nn.BatchNorm1d
        )

        self.norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        # 1. Pass through tiny CNN -> shape: (B, 64, 1, 1)
        features = self.backbone(x)
        
        # 2. Flatten the spatial dimensions -> shape: (B, 64)
        pooled_output = torch.flatten(features, 1)
        
        # 3. Project to latent space
        projected_output = self.projection(pooled_output)

        # 4. Final normalization
        projected_output = self.norm(projected_output)
        
        return projected_output


class DINOv3VisionEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        model_name: str,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        from transformers import AutoImageProcessor, AutoModel

        self.latent_dim = latent_dim
        self.freeze_backbone = freeze_backbone

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        self.register_buffer(
            "image_mean",
            torch.tensor(processor.image_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(processor.image_std).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        dino_dim = self.backbone.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.image_mean) / self.image_std

        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(pixel_values=x).pooler_output
        else:
            features = self.backbone(pixel_values=x).pooler_output

        return self.norm(self.projection(features))


class ActionEncoder(nn.Module):
    def __init__(self, action_space, hidden_dim, output_dim):
        super().__init__()

        action_dim = action_space.shape[0]
        self.mlp = SwiGLUMLP(
            input_dim = action_dim, 
            hidden_dim = hidden_dim,
            output_dim = output_dim
        )

    def forward(self, action):
        return self.mlp(action)

        
class Reward(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.mlp = SwiGLUMLP(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=1
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                init_linear_orthogonal(module, gain=np.sqrt(2))
                
        final_layer = self._get_final_layer()
        if final_layer is not None:
            init_linear_orthogonal(final_layer, gain=0.01)

    def _get_final_layer(self):
        modules = list(self.mlp.modules())
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        return linear_layers[-1] if linear_layers else None

    def forward(self, obs):
        reward = self.mlp(obs)
        return reward
    

class Termination(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.mlp = SwiGLUMLP(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=1
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                init_linear_orthogonal(module, gain=np.sqrt(2))
                
        final_layer = self._get_final_layer()
        if final_layer is not None:
            init_linear_orthogonal(final_layer, gain=0.01)

    def _get_final_layer(self):
        modules = list(self.mlp.modules())
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        return linear_layers[-1] if linear_layers else None

    def forward(self, obs):
        termination = self.mlp(obs)
        return termination


class VisionDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, out_channels=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, hidden_dim // 2)
        self.act1 = nn.SiLU()
        
        self.deconv2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, hidden_dim // 4)
        self.act2 = nn.SiLU()
        
        self.deconv3 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(8, hidden_dim // 8)
        self.act3 = nn.SiLU()
        
        self.deconv4 = nn.ConvTranspose2d(hidden_dim // 8, out_channels, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (..., latent_dim)
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, x.size(-1))
        
        h = self.fc(x)
        h = h.view(-1, self.hidden_dim, 4, 4)
        
        h = self.act1(self.norm1(self.deconv1(h)))
        h = self.act2(self.norm2(self.deconv2(h)))
        h = self.act3(self.norm3(self.deconv3(h)))
        out = self.sigmoid(self.deconv4(h))
        
        out = out.view(*orig_shape, out.size(-3), out.size(-2), out.size(-1))
        return out

        