import torch.nn as nn
import torchvision.models as models

class ImageModel(nn.Module):

    def __init__(self, flen, dropout_rate, intermediate_rep=128):
        super(ImageModel, self).__init__()
        self.feature_length = flen
        resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

        self.model = nn.Sequential(
            nn.Linear(512, intermediate_rep),
            nn.BatchNorm1d(intermediate_rep),
        )

    def forward(self, features):
        image = self.resnet18(features)
        image = image.view(features.shape[0], -1)
        return self.model(image)
