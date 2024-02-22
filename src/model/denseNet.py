import torch
import torch.nn as nn
from torchsummary import summary
import torchsnooper
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.25),
            nn.Sequential(*self._make_dense_layers(64, 12)),
            nn.BatchNorm2d(448),
            nn.ReLU(inplace=True),
            nn.Conv2d(448, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 53 * 53, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    # @torchsnooper.snoop()
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_dense_layers(self, in_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(Bottleneck(in_channels))
            in_channels += 32
        return layers


# 定义Bottleneck模块
class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU(inplace=True)(out)
        out = torch.cat((x, out), 1)
        return out

if __name__ == '__main__':
    net = DenseNet().to(device='cuda')
    tensor = torch.randn(1, 3, 224, 224)
    summary(net, (3, 224, 224))