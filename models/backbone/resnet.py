import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'deformable_resnet18', 'deformable_resnet50',
           'resnet152','u_resnet18', 'shuffle_resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None, shuffle=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.with_shuffle = shuffle is not None
        if not self.with_shuffle:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else :
            self.conv1 = ShuffleAndGhost(inplanes, planes ,stride=stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            from torchvision.ops import DeformConv2d
            deformable_groups = dcn.get('deformable_groups', 1)
            offset_channels = 18 #27
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, kernel_size=3, padding=1)
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
            # offset_mask = self.conv2_offset(out)
            # offset = offset_mask[:, :18, :, :]
            # mask = offset_mask[:, -9:, :, :].sigmoid()
            # out = self.conv2(out, offset, mask)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Up_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None, isUpsample=False, shuffle=None):
        super(Up_BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.isUpsample = isUpsample
        if isUpsample:
            self.upSample1 = nn.Upsample(scale_factor=2, mode='bilinear')#上采样
            self.upSample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            from torchvision.ops import DeformConv2d
            deformable_groups = dcn.get('deformable_groups', 1)
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, kernel_size=3, padding=1)
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.upSample2(x) if self.isUpsample else x

        out = self.conv1(x)
        if self.isUpsample:
            out = self.upSample1(out)#加个上采样
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None, shuffle=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.with_modulated_dcn = False
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            from torchvision.ops import DeformConv2d
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, stride=stride, kernel_size=3, padding=1)
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ShuffleAndGhost(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, ratio=2, dw_size=3, expand_ratio=0.5, stride=1, groups=4):
        super(ShuffleAndGhost, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_size = out_size
        self.groups = groups
        init_channels = math.ceil(out_size / ratio)
        new_channels = init_channels * (ratio - 1)
        hidden_dim = round(init_channels * expand_ratio)

        # ex
        self.conv1 = nn.Conv2d(in_size, hidden_dim, kernel_size=1, stride=1, padding=0, groups=4,bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=4, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # pw
        self.conv3 = nn.Conv2d(hidden_dim, init_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(init_channels)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def channel_shuffle(self, x, groups=4):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups,
                channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.channel_shuffle(out, self.groups)
        out = self.bn2(self.conv2(out))
        out1 = self.bn3(self.conv3(out))
        out2 = self.cheap_operation(out1)
        out = torch.cat([out1, out2], dim=1)
        return out[:, :self.out_size, :, :]


class U_ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, dcn=None):
        self.dcn = dcn
        self.inplanes = 64
        super(U_ResNet, self).__init__()
        self.out_channels = []
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn)

        self.layer_up4 = self._make_layer(block, 512, layers[3], isUpsample=True, dcn=dcn)
        self.layer_up3 = self._make_layer(block, 256, layers[2], isUpsample=True, dcn=dcn)
        self.layer_up2 = self._make_layer(block, 128, layers[1], isUpsample=True, dcn=dcn)
        self.layer_up1 = self._make_layer(block, 64, layers[0], isUpsample=True)

        #保证上采样过程中concat的两个向量一致
        self.align_4_3 = conv3x3(512,256)
        self.align_3_2 = conv3x3(256,128)
        self.align_2_1 = conv3x3(128,64)

        #调整outchannels为[64,128,256,512]
        self.out_channels=self.out_channels[4:]
        self.out_channels.reverse()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1,isUpsample=False, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,isUpsample=isUpsample, dcn=dcn))
        self.out_channels.append(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        y4 = self.layer_up4(x5)
        y4_3=self.align_4_3(y4)
        y3 = self.layer_up3(torch.cat((y4_3, x4),dim=1))
        y3_2=self.align_3_2(y3)
        y2 = self.layer_up2(torch.cat((y3_2, x3),dim=1))
        y2_1=self.align_2_1(y2)
        y1 = self.layer_up1(torch.cat((y2_1, x2),dim=1))

        return y1, y2, y3, y4


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, dcn=None, shuffle=None):
        self.dcn = dcn
        self.shuffle=shuffle
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.out_channels = []
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shuffle=shuffle)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn, shuffle=shuffle)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn, shuffle=shuffle)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn, shuffle=shuffle)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None, shuffle=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn, shuffle=shuffle))
        self.out_channels.append(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5





def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        print('load from imagenet')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def shuffle_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], shuffle=dict(shuffle=1),**kwargs)
    if pretrained:
        pass
    return model


def deformable_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], dcn=dict(deformable_groups=1), **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        print('load from imagenet')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], dcn=dict(deformable_groups=1), **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model

def u_resnet18(pretrained=False, **kwargs):

    model = U_ResNet(Up_BasicBlock, [2,2,2,2], **kwargs)
    if pretrained:
        pass
    return model


if __name__ == '__main__':
    import torch
    from torchsummaryX import summary

    x = torch.zeros(2, 3, 640, 640)
    net = u_resnet18(pretrained=False)
    y = net(x)
    for u in y:
        print(u.shape)
    # summary(net,x)
    print(net.out_channels)
