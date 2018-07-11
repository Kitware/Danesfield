import torch.nn as nn


class SimpleBlock(nn.Module):
    """ conv -> BN -> ReLU -> conv -> BN -> ?down sample? -> sum -> ReLU """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SimpleBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        mid = self.conv1(x)
        mid = self.bn1(mid)
        mid = self.relu(mid)
        mid = self.conv2(mid)
        mid = self.bn2(mid)

        if(self.downsample is not None):
            residual = self.downsample(x)

        mid += residual
        out = self.relu(mid)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layer, num_classes=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(
            kernel_size=3, stride=1, padding=0)

        self.layer1 = self._make_layer(block, 64, layer[0])
        self.layer2 = self._make_layer(block, 128, layer[1], stride=3)
        self.layer3 = self._make_layer(block, 256, layer[2], stride=3)
        self.layer4 = self._make_layer(block, 512, layer[3], stride=3)

        # This makes things fit into the linear part
        self.avgpool = nn.AvgPool1d(2)
        self.avgpool_2 = nn.AvgPool1d(3)
        self.fc1 = nn.Linear(512*block.expansion, num_classes)

        self.soft_max = nn.Softmax(dim=1)

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

        if(mid.size()[2] == 2):
            mid = self.avgpool(mid)
        else:
            mid = self.avgpool_2(mid)
        mid = mid.view(x.size(0), -1)
        mid = self.fc1(mid)
        out = self.soft_max(mid)

        return out


def resnet18(num_classes):
    model = ResNet(SimpleBlock, [2, 2, 2, 2], num_classes)
    return model


def resnet34(num_classes):
    model = ResNet(SimpleBlock, [3, 4, 6, 3], num_classes)
    return model
