import torch
import torch.nn.functional as F
from torchviz import make_dot

from PIL import Image
from models.basemodel import BaseModelAttend, BaseModel
from models.imagemodel import  ImageModel
import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from features.generateFeatures import smile_to_smile_to_image

import pandas as pd

if __name__ == '__main__':





    model = BaseModel(942, 0.1, featureModel=ImageModel, flen=23, return_attns=True)
    model.eval()
    y = torch.zeros(1, 942, dtype=torch.float, requires_grad=False)
    x = torch.zeros(1, 3, 128, 128, dtype=torch.float, requires_grad=False)
    out, attn = model(y,x)
    attn = torch.nn.functional.upsample(attn, size=(128,128))
    plt.imshow(np.transpose(attn.squeeze(0).repeat((3,1,1)).detach().numpy(), (1, 2, 0)), interpolation='nearest')
    plt.show()
    print(attn.shape)
