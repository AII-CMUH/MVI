import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, model_urls, BasicBlock
from torchvision.models.utils import load_state_dict_from_url

class ResNetCf(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNetCf, self).__init__(*args, **kwargs)
        self.cf_fc = None

    def forward(self, x, x2):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x2 = self.cf_fc(x2)
        x = torch.cat((x, x2), 1)
        x = self.fc(x)

        return x

def _resnetcf(arch, block, layers, imagenet_pretrained, progress, **kwargs):
    model = ResNetCf(block, layers, **kwargs)
    if imagenet_pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnetcf18(imagenet_pretrained=False, progress=True, **kwargs):
    model = _resnetcf('resnet18', BasicBlock, [2, 2, 2, 2], imagenet_pretrained,
                      progress, **kwargs)
    model.cf_fc = nn.Sequential(nn.Linear(9, 9), nn.ReLU())
    model.fc = nn.Linear(model.fc.in_features + 9, 2)
    return model
