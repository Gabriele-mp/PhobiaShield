import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FPNNeck(nn.Module):
    def __init__(self, backbone_channels=[128, 256, 512], fpn_channels=256):
        super().__init__()
        self.lateral_c3 = nn.Conv2d(backbone_channels[0], fpn_channels, 1)
        self.lateral_c4 = nn.Conv2d(backbone_channels[1], fpn_channels, 1)
        self.lateral_c5 = nn.Conv2d(backbone_channels[2], fpn_channels, 1)
        
        self.smooth_p3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
    
    def forward(self, features):
        c3, c4, c5 = features
        
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        
        # Smooth
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)
        
        return p3, p4, p5

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_boxes, num_classes):
        super().__init__()
        output_channels = num_boxes * (5 + num_classes)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, output_channels, 1)
        )
    
    def forward(self, x):
        return self.conv(x)

class PhobiaNetFPN_v2(nn.Module):
    """
    Versione V2 Allineata con il Training del compagno.
    """
    def __init__(self, num_classes=5, num_boxes=2):
        super().__init__()
        # Backbone esplicita come da snippet del compagno
        self.conv1 = ConvBlock(3, 16, stride=2)
        self.conv2 = ConvBlock(16, 32, stride=2)
        self.conv3 = ConvBlock(32, 64, stride=2)
        self.conv4 = ConvBlock(64, 128, stride=1)
        self.conv5 = ConvBlock(128, 256, stride=2)
        self.conv6 = ConvBlock(256, 512, stride=2)
        
        self.fpn = FPNNeck([128, 256, 512], 256)
        
        self.head_p3 = DetectionHead(256, num_boxes, num_classes)
        self.head_p4 = DetectionHead(256, num_boxes, num_classes)
        self.head_p5 = DetectionHead(256, num_boxes, num_classes)
    
    def forward(self, x):
        # Forward esplicito
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        c3 = self.conv4(x)  # 128
        c4 = self.conv5(c3) # 256
        c5 = self.conv6(c4) # 512
        
        p3, p4, p5 = self.fpn((c3, c4, c5))
        
        # Permute per avere (B, H, W, C) pronto per il detector
        return (self.head_p3(p3).permute(0, 2, 3, 1), 
                self.head_p4(p4).permute(0, 2, 3, 1), 
                self.head_p5(p5).permute(0, 2, 3, 1))

# Alias per compatibilità se servisse
PhobiaNetFPN = PhobiaNetFPN_v2