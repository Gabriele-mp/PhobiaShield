"""
PhobiaNet - Custom YOLO-style Object Detection Network

Responsabilità Membro B: Model Architect

OPTIMIZED VERSION with:
- ResidualBlock (skip connections)
- CBAM Attention (Channel + Spatial)
- Improved gradient flow
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Architecture:
        x -> Conv -> BN -> LeakyReLU -> Conv -> BN -> (+) -> LeakyReLU
        |______________________________________________|
                        (skip connection)
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for first conv (default 1)
        downsample: If True, use 1x1 conv for skip connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        leaky_slope: float = 0.1
    ):
        super().__init__()
        
        # Main path: Conv -> BN -> LeakyReLU -> Conv -> BN
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity or 1x1 conv if dimensions change)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Skip connection!
        out = self.leaky_relu(out)
        
        return out


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (part of CBAM).
    
    Learns to emphasize important channels.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling branch
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out)
        
        return x * out  # Channel-wise multiplication


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (part of CBAM).
    
    Learns to emphasize important spatial locations.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2,  # avg + max pooling along channel dim
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dim
        out = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        out = self.sigmoid(self.conv(out))
        
        return x * out  # Spatial-wise multiplication


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Sequentially applies channel and spatial attention.
    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
    
    Args:
        in_channels: Input channels
        reduction: Channel reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv2d -> BatchNorm -> LeakyReLU (-> MaxPool)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = "leaky_relu",
        leaky_slope: float = 0.1,
        use_pool: bool = False,
    ):
        super().__init__()
        
        layers = []
        
        # Convolution
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_bn,  # No bias if using BatchNorm
            )
        )
        
        # Batch Normalization
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        if activation == "leaky_relu":
            layers.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Max Pooling
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PhobiaNet(nn.Module):
    """
    Tiny-YOLO style network for phobia object detection.
    
    OPTIMIZED VERSION with:
    - ResidualBlock support (use_residual=True)
    - CBAM attention (use_attention=True)
    
    Architecture:
        Input (3, 416, 416)
        -> ResBlock(16) -> Pool    -> (16, 208, 208)
        -> ResBlock(32) -> Pool    -> (32, 104, 104)
        -> ResBlock(64) -> Pool    -> (64, 52, 52)
        -> ResBlock(128) -> Pool   -> (128, 26, 26)
        -> ResBlock(256) -> Pool   -> (256, 13, 13)
        -> ResBlock(512) + CBAM    -> (512, 13, 13)
        -> Output Conv             -> (B*(5+C), 13, 13)
    
    Args:
        config: Model configuration dictionary
        use_residual: Use ResidualBlock instead of ConvBlock (default: True)
        use_attention: Add CBAM attention after last layer (default: True)
    """
    
    def __init__(
        self,
        config: Dict,
        use_residual: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.config = config
        self.num_classes = config["output"]["num_classes"]
        self.grid_size = config["architecture"]["grid_size"]
        self.num_boxes = config["architecture"]["num_boxes_per_cell"]
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Output channels: B * (5 + C)
        # 5 = [x, y, w, h, confidence]
        self.output_channels = self.num_boxes * (5 + self.num_classes)
        
        # Build backbone
        self.backbone = self._build_backbone(config["architecture"]["layers"])
        
        # CBAM attention on last feature map (OPTIONAL)
        last_channels = config["architecture"]["layers"][-1]["filters"]
        if self.use_attention:
            self.attention = CBAM(last_channels, reduction=16, kernel_size=7)
        
        # Detection head
        self.detection_head = nn.Conv2d(
            last_channels,
            self.output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        # Initialize weights
        self._initialize_weights(config["init"])
    
    def _build_backbone(self, layer_configs: List[Dict]) -> nn.Sequential:
        """Build CNN backbone from layer configurations."""
        layers = []
        in_channels = self.config["architecture"]["in_channels"]
        
        for i, layer_config in enumerate(layer_configs):
            out_channels = layer_config["filters"]
            use_pool = layer_config.get("pool", False)
            
            if self.use_residual:
                # Use ResidualBlock
                stride = 2 if use_pool else 1  # Pool via stride
                layers.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        leaky_slope=self.config["architecture"]["leaky_relu_slope"]
                    )
                )
            else:
                # Use original ConvBlock
                kernel_size = layer_config.get("kernel_size", 3)
                stride = layer_config.get("stride", 1)
                padding = layer_config.get("padding", 1)
                use_bn = layer_config.get("batch_norm", True)
                activation = layer_config.get("activation", "leaky_relu")
                
                layers.append(
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        use_bn=use_bn,
                        activation=activation,
                        leaky_slope=self.config["architecture"]["leaky_relu_slope"],
                        use_pool=use_pool,
                    )
                )
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, init_config: Dict):
        """Initialize model weights."""
        init_type = init_config.get("type", "kaiming")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode=init_config.get("mode", "fan_out"),
                        nonlinearity="leaky_relu"
                    )
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == "normal":
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Output tensor of shape (B, S, S, B*(5+C))
            where S=grid_size, B=num_boxes, C=num_classes
        """
        # Backbone
        features = self.backbone(x)  # (B, 512, 13, 13)
        
        # CBAM attention (if enabled)
        if self.use_attention:
            features = self.attention(features)
        
        # Detection head
        output = self.detection_head(features)  # (B, B*(5+C), 13, 13)
        
        # Reshape to (B, S, S, B*(5+C))
        B = output.size(0)
        output = output.permute(0, 2, 3, 1)  # (B, 13, 13, B*(5+C))
        
        return output
    
    def predict(self, x: torch.Tensor, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Make predictions with confidence thresholding.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            conf_threshold: Confidence threshold
        
        Returns:
            List of predictions for each image in batch
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)  # (B, S, S, B*(5+C))
        
        predictions = []
        batch_size = output.size(0)
        
        for b in range(batch_size):
            batch_preds = []
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for box_idx in range(self.num_boxes):
                        # Get box predictions
                        box_start = box_idx * 5
                        x_cell = output[b, j, i, box_start + 0]
                        y_cell = output[b, j, i, box_start + 1]
                        w = output[b, j, i, box_start + 2]
                        h = output[b, j, i, box_start + 3]
                        conf = torch.sigmoid(output[b, j, i, box_start + 4])
                        
                        # Get class predictions
                        class_start = self.num_boxes * 5
                        class_probs = torch.softmax(
                            output[b, j, i, class_start:class_start + self.num_classes],
                            dim=0
                        )
                        class_id = torch.argmax(class_probs).item()
                        class_prob = class_probs[class_id].item()
                        
                        # Combined confidence
                        final_conf = conf.item() * class_prob
                        
                        if final_conf > conf_threshold:
                            # Convert cell-relative coords to absolute
                            x_abs = (i + torch.sigmoid(x_cell).item()) / self.grid_size
                            y_abs = (j + torch.sigmoid(y_cell).item()) / self.grid_size
                            w_abs = w.item()
                            h_abs = h.item()
                            
                            batch_preds.append({
                                "class_id": class_id,
                                "confidence": final_conf,
                                "bbox": [x_abs, y_abs, w_abs, h_abs],  # [cx, cy, w, h]
                            })
            
            predictions.append(batch_preds)
        
        return predictions


def create_model(config: Dict, use_residual: bool = True, use_attention: bool = True) -> PhobiaNet:
    """
    Create PhobiaNet from config.
    
    Args:
        config: Model configuration
        use_residual: Use ResidualBlock (default: True)
        use_attention: Use CBAM attention (default: True)
    """
    return PhobiaNet(config, use_residual=use_residual, use_attention=use_attention)


if __name__ == "__main__":
    # Test model
    from omegaconf import OmegaConf
    
    # Load config
    config = OmegaConf.load("cfg/model/tiny_yolo_5class.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    
    print("="*60)
    print("Testing PhobiaNet with ResNet + CBAM")
    print("="*60)
    
    # Test 1: With ResNet + CBAM (NEW!)
    print("\n1. Model WITH ResNet + CBAM:")
    model_res = create_model(config, use_residual=True, use_attention=True)
    x = torch.randn(2, 3, 416, 416)
    output = model_res(x)
    params_res = sum(p.numel() for p in model_res.parameters())
    
    print(f"   Input:  {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Params: {params_res:,} ({params_res*4/1e6:.2f} MB)")
    
    # Test 2: Original (for comparison)
    print("\n2. Model WITHOUT ResNet/CBAM (original):")
    model_orig = create_model(config, use_residual=False, use_attention=False)
    output_orig = model_orig(x)
    params_orig = sum(p.numel() for p in model_orig.parameters())
    
    print(f"   Output: {output_orig.shape}")
    print(f"   Params: {params_orig:,} ({params_orig*4/1e6:.2f} MB)")
    
    # Comparison
    print(f"\n📊 Comparison:")
    print(f"   Parameter increase: +{params_res - params_orig:,} (+{(params_res/params_orig - 1)*100:.1f}%)")
    
    # Test prediction
    predictions = model_res.predict(x, conf_threshold=0.5)
    print(f"\n🎯 Predictions: {[len(p) for p in predictions]} detections")
    
    print("\n✓ All tests passed!")