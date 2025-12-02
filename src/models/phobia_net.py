"""
PhobiaNet - Custom YOLO-style Object Detection Network

Responsabilità Membro B: Model Architect
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple


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
    
    Architecture:
        Input (3, 416, 416)
        -> Conv(16) -> Pool    -> (16, 208, 208)
        -> Conv(32) -> Pool    -> (32, 104, 104)
        -> Conv(64) -> Pool    -> (64, 52, 52)
        -> Conv(128) -> Pool   -> (128, 26, 26)
        -> Conv(256) -> Pool   -> (256, 13, 13)
        -> Conv(512)           -> (512, 13, 13)
        -> Output Conv         -> (B*(5+C), 13, 13)
    
    Args:
        config: Model configuration dictionary
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.num_classes = config["output"]["num_classes"]
        self.grid_size = config["architecture"]["grid_size"]
        self.num_boxes = config["architecture"]["num_boxes_per_cell"]
        
        # Output channels: B * (5 + C)
        # 5 = [x, y, w, h, confidence]
        self.output_channels = self.num_boxes * (5 + self.num_classes)
        
        # Build backbone
        self.backbone = self._build_backbone(config["architecture"]["layers"])
        
        # Detection head
        last_channels = config["architecture"]["layers"][-1]["filters"]
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
            kernel_size = layer_config.get("kernel_size", 3)
            stride = layer_config.get("stride", 1)
            padding = layer_config.get("padding", 1)
            use_bn = layer_config.get("batch_norm", True)
            activation = layer_config.get("activation", "leaky_relu")
            use_pool = layer_config.get("pool", False)
            
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


def create_model(config: Dict) -> PhobiaNet:
    """Create PhobiaNet from config."""
    return PhobiaNet(config)


if __name__ == "__main__":
    # Test model
    from omegaconf import OmegaConf
    
    # Load config
    config = OmegaConf.load("cfg/model/tiny_yolo.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    
    # Create model
    model = create_model(config)
    
    # Test forward pass
    x = torch.randn(2, 3, 416, 416)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test prediction
    predictions = model.predict(x, conf_threshold=0.5)
    print(f"Number of detections: {[len(p) for p in predictions]}")
