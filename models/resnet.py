import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

############# ResNet module taken from the resnet module provided with torchvision ################

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'http://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'http://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels, out_channels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv = nn.Conv2d(512, out_channels, kernel_size=3)
        self.bn = nn.BatchNorm2d(out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv(x)
        x = self.bn(x)

        return x


def resnet18(in_channels, out_channels, pretrained):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels, out_channels)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(in_channels, out_channels, pretrained):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels, out_channels)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(in_channels, out_channels, pretrained):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels, out_channels)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(in_channels, out_channels, pretrained):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_channels, out_channels)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(in_channels, out_channels, pretrained):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], in_channels, out_channels)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model

############# ResNet module taken from the resnet module provided with torchvision ################


'''
Select the resnet version to be used

Arguments:
	pretrained : whether to use pretrained model
	resnet_version : 18 / 34 / 50 / 101 / 152
	in_channels : no. of channels in the input (1 for grayscale image and 3 for color image)
	out_channels : no. of channels in the feature vector (can be either content or pose feature)

Returns:
	model : resnet model with the desired specifications
'''
def select_model(pretrained, resnet_version, in_channels, out_channels):

	if resnet_version == 18:
		model = resnet18(in_channels, out_channels, pretrained)
	elif resnet_version == 34:
		model = resnet34(in_channels, out_channels, pretrained)
	elif resnet_version == 50:
		model = resnet50(in_channels, out_channels, pretrained)
	elif resnet_version == 101:
		model = resnet101(in_channels, out_channels, pretrained)
	elif resnet_version == 152:
		model = resnet152(in_channels, out_channels, pretrained)

	return model

'''
Encoder network based on the ResNet architecture
'''
class Encoder(nn.Module):
	'''
	Arguments:
		in_channels : no. of channels in the input (1 for grayscale image and 3 for color image)
		out_channels : no. of channels in the feature vector (can be either content or pose feature)
		resnet_version : 18 / 34 / 50 / 101 / 152
		pretrained : whether to use pretrained model
	'''
	def __init__(self, in_channels, out_channels, resnet_version=18, pretrained=False):
		super(Encoder, self).__init__()

		self.model = select_model(pretrained, resnet_version, in_channels, out_channels)

	'''
	Computes a forward pass through the specified encoder architecture

	Arguments:
		inp : input to the encoder (generally the original image)

	Returns:
		out : output of the forward pass (can be either the content or pose features)
	'''
	def forward(self, inp):

		out = self.model(inp)

		# None is used in line 281 to maintain uniformity in the encoder
		# network across different architectures like DC-GAN, VGG since
		# the encoder outputs both the feature vectors and the skip
		# connection features at each stage. Here, no skip connections are 
		# desired, hence None is passed at the output
		return out, None 

# testing code
if __name__ == '__main__':
	torch.cuda.set_device(0)
	net = Encoder(3,128,18).cuda()
	inp = Variable(torch.FloatTensor(4,3,256,256)).cuda()
	out, _ = net(inp)
	print ('content after encoder: ', out.shape)

	net = Encoder(3,10,18).cuda()
	out, _ = net(inp)
	print ('pose after encoder: ', out.shape)