import torch.nn as nn
import torchvision.models as models

class ImageModel(nn.Module):

    def __init__(self, flen, dropout_rate, intermediate_rep=128):
        super(ImageModel, self).__init__()
        self.feature_length = flen
        resnet18 = models.resnet34(pretrained=True)
        resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        self.resnet181 = nn.Sequential(*list(resnet18.children())[:5])
        self.resnet182 = nn.Sequential(*list(resnet18.children())[5:])
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=5, padding=2, stride=1),
        )

        self.model = nn.Sequential(
            nn.Linear(512, intermediate_rep),
            nn.BatchNorm1d(intermediate_rep),
        )

    def forward(self, features):
        image = self.resnet181(features)
        attention = self.attention(image)
        attention = nn.functional.softmax(attention.view(attention.shape[0], -1), dim=-1).view(attention.shape)
        attention = attention.repeat([1, 64, 1, 1])

        image = self.resnet182(image * attention)
        image = image.view(features.shape[0], -1)
        return self.model(image), attention
