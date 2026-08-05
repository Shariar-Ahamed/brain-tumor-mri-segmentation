import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp  # type: ignore
    HAS_SMP = True
except ImportError:
    smp = None  # type: ignore
    HAS_SMP = False

class StandardUNet(nn.Module):
    """
    Fallback 2D U-Net implementation for 4-channel input if SMP is unavailable.
    """
    def __init__(self, in_channels=4, out_channels=4):
        super(StandardUNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final_conv(d1)


def build_baseline_model(model_name="ResNet34-UNet", in_channels=4, classes=4):
    """
    Factory function to build baseline models.
    Supports: 'ResNet34-UNet', 'U-Net++', 'DeepLabV3+', 'SegFormer'
    """
    if HAS_SMP:
        if model_name == "ResNet34-UNet":
            return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=classes)
        elif model_name == "U-Net++":
            return smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=classes)
        elif model_name == "DeepLabV3+":
            return smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=classes)
        elif model_name == "SegFormer":
            return smp.Unet(encoder_name="mit_b0", encoder_weights=None, in_channels=in_channels, classes=classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    else:
        # Fallback to standard U-Net
        return StandardUNet(in_channels=in_channels, out_channels=classes)
