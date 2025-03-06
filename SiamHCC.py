"""
SiamHCC: A Siamese Network with Hybrid Context Capture for Chinese Character Quality Assessment
Paper-implemented model architecture with self-attention and channel attention mechanisms
"""

import torch
# from torchsummary import summary
# from torchstat import stat
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """Self-Attention Mechanism for capturing long-range dependencies
        
    Args:
        in_channels (int): Number of input channels
    
    Shape:
        Input: (N, C, H, W)
        Output: (N, C, H, W) (same as input)
    
    """
    def __init__(self, in_channels: int = 1024):
        super().__init__()
        self.in_channels = in_channels
        reduction = in_channels // 8

        # Projections
        self.query = nn.Conv2d(in_channels, reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        # Projections
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        # Attention matrix
        attn = self.softmax(torch.bmm(q, k) / math.sqrt(channels // 8))

        # # Apply attention to values
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(batch_size, channels, height, width)

        return self.gamma * out + x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise feature recalibration

    Args:
        in_channels (int): Input feature map channels
        ratio (int): Reduction ratio for bottleneck
        
    Shape:
        Input: (N, C, H, W)
        Output: (N, C, H, W) (same as input)

    """
    def __init__(self, in_channels: int, ratio: int = 4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class DenseLayer(nn.Module):
    """Basic building block for DenseNet architecture
    
    Args:
        in_channels (int): Input channels
        
    Shape:
        Input: (N, C_in, H, W)
        Output: (N, C_out, H, W)
    """
    def __init__(self, in_channels, middle_channels=128, out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels, middle_channels, 1),
            torch.nn.BatchNorm2d(middle_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class DenseBlock(torch.nn.Sequential):
    """Dense Block with multiple densely connected layers

    Args:
        layer_num (int): Number of dense layers in the block
        growth_rate (int): Growth rate for each layer
        in_channels (int): Input channels for first layer
    """
    def __init__(self, layer_num: int, growth_rate: int, in_channels: int, middle_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels + i * growth_rate, middle_channels, growth_rate)
            self.add_module('denselayer%d' % (i), layer)


class TransitionBlock(torch.nn.Sequential):
    """Transition block between dense blocks
    
    Args:
        in_channels (int): Input channels
    """
    def __init__(self, in_channels: int):
        super(TransitionBlock, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm2d(in_channels))
        self.add_module('LeakyReLU', torch.nn.LeakyReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv2d(in_channels, in_channels // 2, 3, padding=1))
        self.add_module('Avgpool', torch.nn.AvgPool2d(2))


class SiamHCC(torch.nn.Module):
    """SiamHCC: Siamese Network with Hybrid Context Capture

    Args:
        config (tuple): Layer configuration for dense blocks (default: (6, 12, 24, 16))
        growth_rate (int): Growth rate for dense layers (default: 32)
        init_channels (int): Initial convolution channels (default: 64)
        in_channels (int): Input image channels (default: 3)
        compression (float): Transition layer compression ratio (default: 0.5)
        
    Shape:
        Input: Two RGB images (N, 3, H, W)
        Output: Similarity score (N, 1)
    """
    def __init__(self, config = (6, 12, 24, 16), growth_rate = 32, init_channels = 64, in_channels = 3,
                 middle_channels = 128):
        super(SiamHCC, self).__init__()

        # Initial Convolutional Block
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        current_channels = init_channels

        # --------------------------
        # Stage 1: DenseBlock + SA + Transition + SE
        # --------------------------
        self.stage1 = self._build_stage(
            num_layers=config[0],
            input_channels=current_channels,
            growth_rate=growth_rate,
            middle_channels=middle_channels
        )
        current_channels = self._calculate_output_channels(
            input_channels=current_channels,
            num_layers=config[0],
            growth_rate=growth_rate,
            apply_compression=True
        )

        # --------------------------
        # Stage 2: DenseBlock + SA + Transition + SE 
        # --------------------------
        self.stage2 = self._build_stage(
            num_layers=config[1],
            input_channels=current_channels,
            growth_rate=growth_rate,
            middle_channels=middle_channels
        )
        current_channels = self._calculate_output_channels(
            input_channels=current_channels,
            num_layers=config[1],
            growth_rate=growth_rate,
            apply_compression=True
        )

        # --------------------------
        # Stage 3: DenseBlock + SA + Transition + SE
        # --------------------------
        self.stage3 = self._build_stage(
            num_layers=config[2],
            input_channels=current_channels,
            growth_rate=growth_rate,
            middle_channels=middle_channels
        )
        current_channels = self._calculate_output_channels(
            input_channels=current_channels,
            num_layers=config[2],
            growth_rate=growth_rate,
            apply_compression=True
        )

        # --------------------------
        # Stage 4: DenseBlock + SA (No Transition)
        # --------------------------
        self.stage4_dense = DenseBlock(
            layer_num=config[3],
            growth_rate=growth_rate,
            in_channels=current_channels,
            middle_channels=middle_channels
        )
        current_channels += config[3] * growth_rate
        self.stage4_sa = SelfAttention(in_channels=current_channels)
        
        # --------------------------
        # Final Layers
        # --------------------------
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._build_fc_layers(current_channels)

    def _build_stage(self, num_layers: int, input_channels: int, growth_rate: int, middle_channels: int) -> nn.Sequential:
        """Constructs a processing stage with DenseBlock and attention modules"""
        return nn.Sequential(
            DenseBlock(num_layers, growth_rate, input_channels, middle_channels),
            SelfAttention(in_channels=input_channels + num_layers*growth_rate),
            TransitionBlock(input_channels + num_layers*growth_rate),
            SEBlock((input_channels + num_layers*growth_rate) // 2)
        )

    def _calculate_output_channels(self, input_channels: int, num_layers: int, growth_rate: int, apply_compression: bool) -> int:
        """Calculates output channels after a processing stage"""
        output_channels = input_channels + num_layers * growth_rate
        return output_channels // 2 if apply_compression else output_channels

    def _build_fc_layers(self, input_dim: int) -> nn.Sequential:
        """Constructs the final fully connected layers"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.BatchNorm1d(input_dim//2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(input_dim//2, input_dim//4),
            nn.BatchNorm1d(input_dim//4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(input_dim//4, 1)
        )


    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Processing pipeline for a single input branch"""
        # Initial processing
        x = self.initial_block(x)
        
        # Feature extraction stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Final dense block, pooling and flatten 
        x = self.stage4_dense(x)
        x = self.avgpool(x)
        x = self.stage4_sa(x)

        return torch.flatten(x, 1)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Computes similarity score between two input images."""
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return self.fc(out1 - out2)



