import torch
import torch.nn as nn
import timm


class EfficientUNet(nn.Module):
    def __init__(self, encoder_name='efficientnet_b0', pretrained=False):
        super(EfficientUNet, self).__init__()

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )

        self.up5 = nn.ConvTranspose2d(1280, 256, kernel_size=2, stride=2)
        self.dec5 = self._decoder_block(256 + 112, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(128 + 40, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(64 + 24, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(32 + 16, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(16, 16)

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def _decoder_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = self.encoder(x)
        f1, f2, f3, f4, f5 = features[0], features[1], features[2], features[3], features[4]

        x = self.bottleneck(f5)

        x = self.up5(x)
        x = torch.cat([x, f4], dim=1)
        x = self.dec5(x)

        x = self.up4(x)
        x = torch.cat([x, f3], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, f2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        return self.out_conv(x)
