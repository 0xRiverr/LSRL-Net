# import torch
# from torch import nn
# from .resnet import resnet34
#
#
# class ConvBlock(nn.Module):
#     def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
#         super(ConvBlock, self).__init__()
#
#         ops = []
#         for i in range(n_stages):
#             if i==0:
#                 input_channel = n_filters_in
#             else:
#                 input_channel = n_filters_out
#
#             ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
#             if normalization == 'batchnorm':
#                 ops.append(nn.BatchNorm3d(n_filters_out))
#             elif normalization == 'groupnorm':
#                 ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
#             elif normalization == 'instancenorm':
#                 ops.append(nn.InstanceNorm3d(n_filters_out))
#             elif normalization != 'none':
#                 assert False
#             ops.append(nn.ReLU(inplace=True))
#
#         self.conv = nn.Sequential(*ops)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
# class ResidualConvBlock(nn.Module):
#     def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
#         super(ResidualConvBlock, self).__init__()
#
#         ops = []
#         for i in range(n_stages):
#             if i == 0:
#                 input_channel = n_filters_in
#             else:
#                 input_channel = n_filters_out
#
#             ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
#             if normalization == 'batchnorm':
#                 ops.append(nn.BatchNorm3d(n_filters_out))
#             elif normalization == 'groupnorm':
#                 ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
#             elif normalization == 'instancenorm':
#                 ops.append(nn.InstanceNorm3d(n_filters_out))
#             elif normalization != 'none':
#                 assert False
#
#             if i != n_stages-1:
#                 ops.append(nn.ReLU(inplace=True))
#
#         self.conv = nn.Sequential(*ops)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = (self.conv(x) + x)
#         x = self.relu(x)
#         return x
#
# class DownsamplingConvBlock(nn.Module):
#     def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
#         super(DownsamplingConvBlock, self).__init__()
#
#         ops = []
#         if normalization != 'none':
#             ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
#             if normalization == 'batchnorm':
#                 ops.append(nn.BatchNorm3d(n_filters_out))
#             elif normalization == 'groupnorm':
#                 ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
#             elif normalization == 'instancenorm':
#                 ops.append(nn.InstanceNorm3d(n_filters_out))
#             else:
#                 assert False
#         else:
#             ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
#
#         ops.append(nn.ReLU(inplace=True))
#
#         self.conv = nn.Sequential(*ops)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
# class UpsamplingDeconvBlock(nn.Module):
#     def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
#         super(UpsamplingDeconvBlock, self).__init__()
#
#         ops = []
#         if normalization != 'none':
#             ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
#             if normalization == 'batchnorm':
#                 ops.append(nn.BatchNorm3d(n_filters_out))
#             elif normalization == 'groupnorm':
#                 ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
#             elif normalization == 'instancenorm':
#                 ops.append(nn.InstanceNorm3d(n_filters_out))
#             else:
#                 assert False
#         else:
#             ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
#
#         ops.append(nn.ReLU(inplace=True))
#
#         self.conv = nn.Sequential(*ops)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
# class Upsampling(nn.Module):
#     def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
#         super(Upsampling, self).__init__()
#
#         ops = []
#         ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
#         ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
#         if normalization == 'batchnorm':
#             ops.append(nn.BatchNorm3d(n_filters_out))
#         elif normalization == 'groupnorm':
#             ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
#         elif normalization == 'instancenorm':
#             ops.append(nn.InstanceNorm3d(n_filters_out))
#         elif normalization != 'none':
#             assert False
#         ops.append(nn.ReLU(inplace=True))
#
#         self.conv = nn.Sequential(*ops)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
# class C4_head(nn.Module):
#     def __init__(self,in_channel=256,out_channel=512):
#         super(C4_head, self).__init__()
#
#         self.conv1 = nn.Conv3d(in_channel,out_channel, kernel_size=(3,3,3), stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm3d(out_channel)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 2), stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm3d(out_channel)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv3d(out_channel, out_channel*2, kernel_size=(2,2,1), stride=1, padding=0, bias=False)
#
#     def forward(self, x, bs):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         bs_num,c,w,h,d = x.shape
#         x= torch.reshape(x,(bs,bs_num//bs,c*w*h*d))
#         return x
#
# class C5_head(nn.Module):
#     def __init__(self,in_channel=512,out_channel=1024):
#         super(C5_head, self).__init__()
#
#
#         self.conv1 = nn.Conv3d(in_channel,out_channel, kernel_size=(3,3,2), stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm3d(out_channel)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
#
#     def forward(self, x, bs):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         bs_num, c, w, h, d = x.shape
#         x = torch.reshape(x, (bs, bs_num // bs, c * w * h * d))
#         return x
#
# class Resnet34(nn.Module):
#     def __init__(self, resnet_encoder=None, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
#         super(Resnet34, self).__init__()
#         self.has_dropout = has_dropout
#         self.resnet_encoder = resnet34()
#
#         self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)
#
#         self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
#
#         self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
#
#         self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
#
#         self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
#         self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
#
#         self.dropout = nn.Dropout3d(p=0.5, inplace=False)
#
#         self.__init_weight()
#
#
#     def decoder(self, features):
#         x1 = features[0]
#         x2 = features[1]
#         x3 = features[2]
#         x4 = features[3]
#         x5 = features[4]
#
#         x5_up = self.block_five_up(x5)
#         x5_up = x5_up + x4
#
#         x6 = self.block_six(x5_up)
#         x6_up = self.block_six_up(x6)
#         x6_up = x6_up + x3
#
#         x7 = self.block_seven(x6_up)
#         x7_up = self.block_seven_up(x7)
#         x7_up = x7_up + x2
#
#         x8 = self.block_eight(x7_up)
#         x8_up = self.block_eight_up(x8)
#         x8_up = x8_up + x1
#         x9 = self.block_nine(x8_up)
#         # x9 = F.dropout3d(x9, p=0.5, training=True)
#         if self.has_dropout:
#             x9 = self.dropout(x9)
#         out = self.out_conv(x9)
#         return out
#
#     def forward(self, input, batch_size=4):
#         resnet_features = self.resnet_encoder(input)
#         out = self.decoder(resnet_features)
#         return out
#
#     def __init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

import torch
import torch.nn as nn
from torch.nn import functional as F


class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)

class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class Resnet34(nn.Module):
    def __init__(self, classes_num):
        super(Resnet34, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, classes_num)
        )

    def forward(self, x):
        x = self.prepare(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

