import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = EncoderBlock(3, 16, False)
        self.encoder2 = EncoderBlock(16, 32, True)
        self.encoder3 = EncoderBlock(32, 64, True)
        self.encoder4 = EncoderBlock(64, 128, True)
        self.encoder5 = EncoderBlock(128, 256, True)
        self.encoder6 = EncoderBlock(256, 512, True)
        self.encoder7 = EncoderBlock(512, 1024, True)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        x6 = self.encoder6(x5)
        x7 = self.encoder7(x6)
        return x1, x2, x3, x4, x5, x6, x7


class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.conv3 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super().__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder1 = DecoderBlock(1024, 512, 512)
        self.decoder2 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder4 = DecoderBlock(128, 64, 64)
        self.decoder5 = DecoderBlock(64, 32, 32)
        self.decoder6 = DecoderBlock(32, 16, 16)

        self.conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 1, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        x = self.decoder1(x6, x7)
        x = self.decoder2(x5, x)
        x = self.decoder3(x4, x)
        x = self.decoder4(x3, x)
        x = self.decoder5(x2, x)
        x = self.decoder6(x1, x)

        x = self.relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()

    def forward(self, x):
        x1, x2, x3, x4, x5, x6, x7 = self.encoder(x)
        x7 = self.bottleneck(x7)
        x = self.decoder(x1, x2, x3, x4, x5, x6, x7)
        return x
