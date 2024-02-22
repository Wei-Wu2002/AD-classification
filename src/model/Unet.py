# import torch
# import torch.nn as nn
# from torchsummary import summary
#
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             DoubleConv(in_channels, 64),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             DoubleConv(64, 128),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             DoubleConv(128, 256),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             DoubleConv(256, 512),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
#             DoubleConv(512, 256),
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#             DoubleConv(256, 128),
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
#             DoubleConv(128, 64),
#             nn.Conv2d(64, out_channels, kernel_size=1)
#         )
#
#     # @torchsnooper.snoop()
#     def forward(self, x):
#         x1 = self.encoder[0](x)
#         x2 = self.encoder[1](x1)
#         x3 = self.encoder[2](x2)
#         x4 = self.encoder[3](x3)
#         x5 = self.encoder[4](x4)
#         x6 = self.encoder[5](x5)
#         x7 = self.encoder[6](x6)
#
#         x = self.decoder[0](x7)
#         x = torch.cat([x, x5], dim=1)
#         x = self.decoder[1](x)
#         x = self.decoder[2](x)
#         x = torch.cat([x, x3], dim=1)
#         x = self.decoder[3](x)
#         x = self.decoder[4](x)
#         x = torch.cat([x, x1], dim=1)
#         x = self.decoder[5](x)
#         x = self.decoder[6](x)
#
#         return x
#
# class UNetWithClassifier(nn.Module):
#     def __init__(self, in_channels, out_channels, num_classes):
#         super(UNetWithClassifier, self).__init__()
#         self.unet = UNet(in_channels, out_channels)
#         self.classifier = nn.Sequential(
#             nn.Linear(out_channels * 224 * 224, 256),  # Assuming the output size of the UNet before the classifier is 512x7x7
#             nn.ReLU(inplace=True),
#             nn.Linear(256, num_classes)
#         )
#
#     def forward(self, x):
#         features = self.unet(x)
#         features = features.view(features.size(0), -1)  # Flatten the feature map
#         output = self.classifier(features)
#         return output

# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             DoubleConv(in_channels, 64),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             DoubleConv(64, 128),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             DoubleConv(128, 256),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             DoubleConv(256, 512),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
#             DoubleConv(512, 256),
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#             DoubleConv(256, 128),
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
#             DoubleConv(128, 64),
#             nn.Conv2d(64, out_channels, kernel_size=1)
#         )
#
#     @torchsnooper.snoop()
#     def forward(self, x):
#         x1 = self.encoder[0](x)
#         x2 = self.encoder[1](x1)
#         x3 = self.encoder[2](x2)
#         x4 = self.encoder[3](x3)
#         x5 = self.encoder[4](x4)
#         x6 = self.encoder[5](x5)
#         x7 = self.encoder[6](x6)
#
#         x = self.decoder[0](x7)
#         x = torch.cat([x, x5], dim=1)
#         x = self.decoder[1](x)
#         x = self.decoder[2](x)
#         x = torch.cat([x, x3], dim=1)
#         x = self.decoder[3](x)
#         x = self.decoder[4](x)
#         x = torch.cat([x, x1], dim=1)
#         x = self.decoder[5](x)
#         x = self.decoder[6](x)
#         x = torch.cat([x, x1], dim=1)
#         x = self.decoder[7](x)
#
#         return x

# # 创建UNet模型实例
# unet = UNet(in_channels=3, out_channels=2)  # 输入通道数为3，输出通道数为2（二分类问题）
#
# # 定义一个随机输入
# dummy_input = torch.randn(1, 3, 208, 179)
#
# # 打印模型结构和参数数量
# summary(unet, (3, 208, 179), batch_size=1, device='cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F


# import matplotlib
# from attention import ChannelAttention


# UNet
# 把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # GaborConv2d(in_ch, out_ch, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(out_ch),  # 添加了BN
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # GaborConv2d(out_ch, out_ch, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        # self.gconv = GaborConv2d(in_ch, in_ch*32, kernel_size = (3, 3), padding =1)
        self.conv1 = DoubleConv(in_ch, 64)
        # self.conv1 = nn.Conv2d(in_ch*32, 64, 3, padding = 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # self.attention_layer = ChannelAttention(1024)
        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, 2, 1)  # 64,2,1
        # self.fc1 = nn.Linear(2 * 224 * 224, 1024)  # 530*50 = 26500
        # self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 256)
        self.fc3 = nn.Linear(256, out_ch)
        # self.fc4 = nn.Linear(64, out_ch)

        self.mlp = nn.Sequential(
            nn.Linear(2 * 224 * 224, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_ch),
            nn.Sigmoid()
        )

    # @torchsnooper.snoop()
    def forward(self, x):
        # g1 = self.conv1(x)
        c1 = self.conv1(x)  # [16, 64, 446, 50]
        p1 = self.pool1(c1)  # [16, 64, 223, 25]
        c2 = self.conv2(p1)  # [16, 128, 223, 25]
        p2 = self.pool2(c2)  # [16, 128, 111, 12]
        c3 = self.conv3(p2)  # [16, 256, 111, 12]
        p3 = self.pool3(c3)  # [16, 256, 55, 6]
        c4 = self.conv4(p3)  # [16, 512, 55, 6]
        p4 = self.pool4(c4)  # [16, 512, 27, 3]
        c5 = self.conv5(p4)  # [16, 1024, 27, 3]
        up_6 = self.up6(c5)  # [16, 512, 54, 6]
        height_padding = c4.size(2) - up_6.size(2)  # 计算高度上的填充量
        # 进行高度填充
        if height_padding > 0:
            up_6 = F.pad(up_6, (0, 0, height_padding, 0))  # 在上方进行填充
        elif height_padding < 0:
            c4 = F.pad(c4, (0, 0, -height_padding, 0))  # 在上方进行填充

        merge6 = torch.cat([up_6, c4], dim=1)  # [16, 1024, 55, 6]
        c6 = self.conv6(merge6)  # [16, 512, 55, 6]
        up_7 = self.up7(c6)  # [16, 256, 110, 12]

        height_padding = c3.size(2) - up_7.size(2)  # 计算高度上的填充量
        # 进行高度填充
        if height_padding > 0:
            up_7 = F.pad(up_7, (0, 0, height_padding, 0))  # 在上方进行填充
        elif height_padding < 0:
            c3 = F.pad(c3, (0, 0, -height_padding, 0))  # 在上方进行填充
        merge7 = torch.cat([up_7, c3], dim=1)  # [16, 512, 111, 12]
        c7 = self.conv7(merge7)  # [16, 256, 132, 12]

        up_8 = self.up8(c7)  # [16, 128, 264, 24]
        pad = nn.ReplicationPad2d(padding=(c2.shape[2] - up_8.shape[2], 0, c2.shape[3] - up_8.shape[3], 0))
        up_8 = pad(up_8)  # [16, 128, 265, 25]
        merge8 = torch.cat([up_8, c2], dim=1)  # [16, 256, 265, 25]
        c8 = self.conv8(merge8)  # [16, 128, 265, 25]
        up_9 = self.up9(c8)  # [16, 64, 530, 50]
        merge9 = torch.cat([up_9, c1], dim=1)  # [16, 128, 530, 50]
        c9 = self.conv9(merge9)  # [16, 128, 530, 50]
        out = self.conv10(c9)  # [16, 2, 530, 50]
        # out = nn.Sigmoid()(c10)  # [16, 2, 530, 50]
        proj = out.view(-1, 2 * 224 * 224)  # 将数据平整为一维的, [16, 53000]
        pred = self.mlp(proj)
        # out = self.fc1(proj)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # pred = self.fc3(out)
        # out = self.relu(out)
        # pred = self.fc4(out)
        return pred
