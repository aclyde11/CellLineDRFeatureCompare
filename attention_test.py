import torch
from train import load_data_models
import torch.nn.functional as F
from torchviz import make_dot
from features.datasets import ImageDatasetOnFly
from PIL import Image
from models.basemodel import BaseModelAttend, BaseModel
from models.imagemodel import  ImageModel
import os
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

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

if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_attn_pred(rnaseq, drugfeats, value):
    rnaseq, drugfeats, value = rnaseq.to(device), drugfeats.to(device), value.to(device)
    model.return_attns = True
    pred, attn = model(rnaseq.unsqueeze(0), drugfeats.unsqueeze(0))
    attn = attn.squeeze(0).detach()
    attn = torch.sum(attn, dim=0, keepdim=True)
    attn = attn.repeat([3, 1, 1]).unsqueeze(0)
    attn = torch.nn.functional.interpolate(attn, size=(128, 128), mode='bicubic')
    # drug_image = torch.cat([drugfeats.unsqueeze(0), 1.0 - attn[:, 1, :, :].unsqueeze(1)], dim=1)
    return pred, attn, drugfeats

if __name__ == '__main__':
    cmap = pl.cm.binary
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    cells, drugs, values, cell_frame, smiles = load_data_models(32, 'cells', 'image', 8, batch_size=16, dropout_rate=0.15, data_escape=True)
    dset = ImageDatasetOnFly(cells, cell_frame, smiles, values, drugs)


    model = torch.load('saved_models/model.pt', map_location='cpu')['inference_model']
    model.eval()

    pred, attn, image = get_attn_pred(*dset[45234])
    print(pred.shape, attn.shape, image.shape)

    plt.imshow(np.transpose(image.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    plt.contourf(list(range(128)), list(range(128)), 1.0 - attn.squeeze(0)[0], cmap=my_cmap, levels=5)
    plt.colorbar()
    # plt.imshow(np.transpose(image.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    # plt.imshow(np.transpose(attn.squeeze(0).numpy(), (1, 2, 0)), interpolation='nearest')
    plt.title("Predicition value " + str(pred.item()))
    plt.show()