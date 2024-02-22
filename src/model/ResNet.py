# import torch
# import torch.nn as nn
# from torchviz import make_dot
# from torchsummary import summary
#
# # 定义残差块
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = None
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return out
#
# # 定义ResNet模型
# class ResNet(nn.Module):
#     def __init__(self, num_classes=2, dropout_rate= 0.5):
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self.make_layer(64, 64, 2, stride=1)
#         self.layer2 = self.make_layer(64, 128, 2, stride=2)
#         self.layer3 = self.make_layer(128, 256, 2, stride=2)
#         self.layer4 = self.make_layer(256, 512, 2, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.fc = nn.Linear(512, num_classes)
#
#     def make_layer(self, in_channels, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(ResidualBlock(in_channels, out_channels, stride))
#             in_channels = out_channels
#         return nn.Sequential(*layers)
#
#     @torchsnooper.snoop()
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x


# # 创建ResNet模型实例
# resnet = ResNet()
#
# # 定义一个随机输入
# dummy_input = torch.randn(1, 3, 208, 179)
#
# summary(resnet, (3, 208, 179), batch_size=1, device='cpu')

# # 使用torchviz生成可视化图
# dot = make_dot(resnet(dummy_input), params=dict(resnet.named_parameters()))
#
# # 保存可视化图到文件
# dot.render(filename='resnet_visualization', format='png')
#
# # 显示可视化图
# dot.view()

import torch.nn as nn
from torch.nn import functional as F


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class ResNet18(nn.Module):
    def __init__(self, device):
        super(ResNet18, self).__init__()
        self.feat_digger = nn.Sequential(
            # GaborConv2d(1, 64, kernel_size=(7, 7), stride=2, padding=3, device=device),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Sequential(RestNetBasicBlock(64, 64, 1), RestNetBasicBlock(64, 64, 1)),

            nn.Sequential(RestNetDownBlock(64, 128, [2, 1]), RestNetBasicBlock(128, 128, 1)),

            nn.Sequential(RestNetDownBlock(128, 256, [2, 1]), RestNetBasicBlock(256, 256, 1)),

            nn.Sequential(RestNetDownBlock(256, 512, [2, 1]), RestNetBasicBlock(512, 512, 1)),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),

            nn.Flatten()
        )

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    # @torchsnooper.snoop()
    def forward(self, x):
        proj = self.feat_digger(x)
        pred = self.mlp(proj)
        # pred = F.relu(self.fc1(proj))
        # pred = self.fc2(pred)
        # pred = F.relu(self.fc2(pred))
        # pred = self.sigmoid(pred)
        return pred