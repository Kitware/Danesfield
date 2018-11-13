###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import torch.nn as nn


class SimpleBlock_1D(nn.Module):
    """ conv -> BN -> ReLU -> conv -> BN -> ?down sample? -> sum -> ReLU """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, p=False):
        super(SimpleBlock_1D, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(
            planes, momentum=0.1, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(
            planes, momentum=0.1, affine=True)
        self.downsample = downsample
        self.stride = stride
        self.p = p

    def forward(self, x):
        residual = x
        mid = self.conv1(x)
        mid = self.bn1(mid)
        mid = self.relu(mid)
        mid = self.conv2(mid)
        mid = self.bn2(mid)

        if(self.downsample is not None):
            residual = self.downsample(x)

        mid += residual  # Add input to block here
        out = self.relu(mid)

        return out


class Model_A(nn.Module):
    def __init__(self, block, layer, num_classes=12):
        self.inplanes = 64
        super(Model_A, self).__init__()
        self.conv1 = nn.Conv1d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64, momentum=0.1, affine=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(
            kernel_size=3, stride=1, padding=0)

        self.layer1 = self._make_layer(block, 64, layer[0], p=True)
        self.layer2 = self._make_layer(block, 128, layer[1], stride=3)
        self.layer3 = self._make_layer(block, 256, layer[2], stride=3)
        self.layer4 = self._make_layer(block, 512, layer[3], stride=3)

        # This makes things fit into the linear part
        self.avgpool = nn.AvgPool1d(2)
        self.avgpool_2 = nn.AvgPool1d(3)
        self.fc1 = nn.Linear(512*block.expansion, num_classes)

        self.UP_MS = nn.Upsample(
            scale_factor=2, mode='linear', align_corners=True)
        self.UP_M = nn.Upsample(
            scale_factor=4, mode='linear', align_corners=True)

    def _make_layer(self, block, planes, blocks, stride=1, p=False):
        downsample = None
        if(stride != 1 or self.inplanes != planes*block.expansion):
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes*block.expansion, momentum=0.1, affine=True))

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, p))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.size()[2] == 8 or x.size()[2] == 11:
            x = self.UP_M(x)
        else:
            x = self.UP_MS(x)
        mid = self.conv1(x)
        mid = self.bn1(mid)
        mid = self.relu(mid)
        mid = self.maxpool(mid)

        mid = self.layer1(mid)
        mid = self.layer2(mid)
        mid = self.layer3(mid)
        mid = self.layer4(mid)

        if(mid.size()[2] == 2):
            mid = self.avgpool(mid)
        else:
            mid = self.avgpool_2(mid)

        mid = mid.view(x.size(0), -1)

        out = self.fc1(mid)

        return out


class Model_B(nn.Module):
    def __init__(self, block, layer, num_classes=12):
        self.inplanes = 64
        super(Model_B, self).__init__()
        self.conv1 = nn.Conv1d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(
            kernel_size=3, stride=1, padding=0)

        self.layer1 = self._make_layer(block, 64, layer[0])
        self.layer2 = self._make_layer(block, 128, layer[1], stride=3)
        self.layer3 = self._make_layer(block, 256, layer[2], stride=3)
        self.layer4 = self._make_layer(block, 512, layer[3], stride=3)

        # This makes things fit into the linear part
        self.avgpool = nn.AvgPool1d(3)
        self.avgpool_2 = nn.AvgPool1d(6)
        self.avgpool_3 = nn.AvgPool1d(5)
        self.avgpool_4 = nn.AvgPool1d(4)
        self.avgpool_5 = nn.AvgPool1d(62)
        self.fc1 = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if(stride != 1 or self.inplanes != planes*block.expansion):
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes*block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        mid = self.conv1(x)
        mid = self.bn1(mid)
        mid = self.relu(mid)
        mid = self.maxpool(mid)

        mid = self.layer1(mid)
        mid = self.layer2(mid)
        mid = self.layer3(mid)
        mid = self.layer4(mid)
        if x.size()[2] == 8*10:
            mid = self.avgpool(mid)
        elif x.size()[2] == 8*20:
            mid = self.avgpool_2(mid)
        elif x.size()[2] == 8*15:
            mid = self.avgpool_3(mid)
        elif x.size()[2] == 8*13:
            mid = self.avgpool_4(mid)
        elif x.size()[2] == 8*208:
            mid = self.avgpool_5(mid)
        mid = mid.view(x.size(0), -1)
        out = self.fc1(mid)

        return out


def model_A(num_classes):
    model = Model_A(SimpleBlock_1D, [2, 2, 2, 2], num_classes)
    return model


def model_B(num_classes):
    model = Model_B(SimpleBlock_1D, [2, 2, 2, 2], num_classes)
    return model
