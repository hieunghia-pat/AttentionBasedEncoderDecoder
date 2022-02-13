from torch import nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self, name, d_model, pretrained=False, dropout=0.5):
        super(FeatureExtractor, self).__init__()

        if name == 'vgg11':
            backbone = models.vgg11_bn(pretrained=pretrained)
            backbone = backbone.features
            final_in_dim = 512
        elif name == 'vgg19':
            backbone = models.vgg19_bn(pretrained=pretrained)
            backbone = backbone.features
            final_in_dim = 512
        elif name == "resnet101":
            backbone = models.resnet101(pretrained=pretrained)
            backbone = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            )
            final_in_dim = 2048
        elif name == "resnet152":
            backbone = models.resnet152(pretrained=pretrained)
            backbone = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            )
            final_in_dim = 2048
        elif name == "resnext101":
            backbone = models.resnext101_32x8d(pretrained=pretrained)
            backbone = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            )
            final_in_dim = 2048
 
        self.backbone = backbone
        self.linear = nn.Conv2d(final_in_dim, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(d_model, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Shape: 
            - x: (bs, dim, h, w)
            - output: (w, bs, dim)
        """
        conv = self.backbone(x)
        conv = self.dropout(self.linear(conv))
        conv = self.relu(self.bn(conv))

        bs, c, h, w = conv.shape
        conv = conv.contiguous().view(bs, c*h, w).permute(2, 0, 1) # (w, bs, dim)

        return conv