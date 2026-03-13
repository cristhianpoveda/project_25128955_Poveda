import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """
        Double convolution + batch normalisation + ReLU
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class HandGestureModel(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        """
        PROPOSED PARALLEL U-NET ARCHITECTURE
        Args:
            in_channels (int): 3 for RGB, 4 for RGB-D.
            n_classes (int): Number of gesture classes (10).
        """
        super(HandGestureModel, self).__init__()
        
        # 3 or 4-channel input
        self.inc = DoubleConv(in_channels, 64)

        # Encoder
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Semantic features map
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (Segmentation)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(1024 + 512, 512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(512 + 256, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = DoubleConv(256 + 128, 128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = DoubleConv(128 + 64, 64)
        
        # Final Prediction
        self.seg_out = nn.Conv2d(64, 1, kernel_size=1) 

        # Classification block
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # overfitting
            nn.Linear(512, n_classes)
        )

    def forward(self, x):

        """
        NEURAL NETWORK FORWARD PASS
        Args:
            x: input tensor

        Returns:
            mask_logits: hand mask logits
            clas_logits: Hand gesture class logits
        """
        
        # Encoder
        x1 = self.inc(x)        # [B, 64, H, W]
        x2 = self.down1(x1)     # [B, 128, H/2, W/2]
        x3 = self.down2(x2)     # [B, 256, H/4, W/4]
        x4 = self.down3(x3)     # [B, 512, H/8, W/8]
        
        x5 = self.bottleneck(nn.MaxPool2d(2)(x4)) # [B, 1024, H/16, W/16]

        # Classification
        class_logits = self.class_head(self.global_pool(x5))

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1) # Skip connections
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x) # high res feature maps

        mask_logits = self.seg_out(x) # [B, 1, H, W]
        
        return mask_logits, class_logits

if __name__ == "__main__":

    # Handling 3 channels
    
    model_rgb = HandGestureModel(in_channels=3)
    out_mask, out_class = model_rgb(torch.randn(1, 3, 256, 256))
    print(f"RGB Input -> Mask: {out_mask.shape}, Class: {out_class.shape}")

    # Handling 4 channels

    model_rgbd = HandGestureModel(in_channels=4)
    out_mask, out_class = model_rgbd(torch.randn(1, 4, 256, 256))
    print(f"RGB-D Input -> Mask: {out_mask.shape}, Class: {out_class.shape}")