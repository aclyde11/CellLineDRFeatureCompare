import pickle

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

from features.generateFeatures import smile_to_mordred, smile_to_smile_to_image, smiles_to_graph


class ImageDatasetOnFly(Dataset):
    def __init__(self, cells, rnaseq, g, values, drugs):
        self.graphs = g
        self.cells = cells
        self.rnaseq = rnaseq
        self.values = values
        self.drugs = drugs

    def __getitem__(self, item):
        t = self.graphs[self.drugs[item]]
        t = smile_to_smile_to_image(t)
        cell_data = np.array(self.rnaseq[self.rnaseq['lincs.Sample'] == self.cells[item]].iloc[0, 1:], dtype=np.float32)
        return torch.from_numpy(cell_data).float(), t.float(), torch.from_numpy(
            self.values[:, item]).float()

    def __len__(self):
        return self.cells.shape[0]


class ImageDataset(Dataset):
    def __init__(self, cells, rnaseq, g, values, drugs):
        self.graphs = g
        self.cells = cells
        self.rnaseq = rnaseq
        self.values = values
        self.drugs = drugs

    def __getitem__(self, item):
        cell_data = np.array(self.rnaseq[self.rnaseq['lincs.Sample'] == self.cells[item]].iloc[0, 1:], dtype=np.float32)
        return torch.from_numpy(cell_data).float(), self.graphs[self.drugs[item]].float(), torch.from_numpy(
            self.values[:, item]).float()

    def __len__(self):
        return self.cells.shape[0]


class VectorDatasetOnFly(Dataset):
    def __init__(self, cells, rnaseq, g, values, drugs, imputer_dict=True):
        self.graphs = g
        self.rnaseq = rnaseq
        self.values = values
        self.drugs = drugs
        self.cells = cells
        if imputer_dict:
            with open("data/imputer.pkl", 'rb') as f:
                self.imputer_dict = pickle.load(f)
        else:
            self.imputer_dict = None

    def __getitem__(self, item):
        t = self.graphs[self.drugs[item]]
        t = smile_to_mordred(t, imputer_dict=self.imputer_dict)
        cell_data = np.array(self.rnaseq[self.rnaseq['lincs.Sample'] == self.cells[item]].iloc[0, 1:], dtype=np.float32)
        return torch.from_numpy(cell_data).float(), torch.from_numpy(t).float(), torch.from_numpy(
            self.values[:, item]).float()

    def __len__(self):
        return self.cells.shape[0]


class VectorDataset(Dataset):
    def __init__(self, cells, rnaseq, g, values, drugs):
        self.graphs = g
        self.rnaseq = rnaseq
        self.values = values
        self.cells = cells
        self.drugs = drugs

    def __getitem__(self, item):
        t = self.graphs[self.drugs[item]]
        cell_data = np.array(self.rnaseq[self.rnaseq['lincs.Sample'] == self.cells[item]].iloc[0, 1:], dtype=np.float32)
        return torch.from_numpy(cell_data).float(), torch.from_numpy(t).float(), torch.from_numpy(
            self.values[:, item]).float()

    def __len__(self):
        return self.cells.shape[0]


class GraphDatasetOnFly(Dataset):
    def __init__(self, cells, rnaseq, g, values, drugs):
        self.graphs = g
        self.cells = cells
        self.rnaseq = rnaseq
        self.values = values
        self.drugs = drugs

    def __getitem__(self, item):
        cell_data = np.array(self.rnaseq[self.rnaseq['lincs.Sample'] == self.cells[item]].iloc[0, 1:], dtype=np.float32)
        t = self.graphs[self.drugs[item]]
        t = smiles_to_graph(t)
        return torch.from_numpy(cell_data), t, torch.from_numpy(self.values[:, item])

    def __len__(self):
        return self.cells.shape[0]


class GraphDataset(Dataset):
    def __init__(self, cells, rnaseq, g, values, drugs):
        self.graphs = g
        self.cells = cells
        self.rnaseq = rnaseq
        self.values = values
        self.drugs = drugs

    def __getitem__(self, item):
        cell_data = np.array(self.rnaseq[self.rnaseq['lincs.Sample'] == self.cells[item]].iloc[0, 1:], dtype=np.float32)
        return torch.from_numpy(cell_data), self.graphs[self.drugs[item]], torch.from_numpy(self.values[:, item])

    def __len__(self):
        return self.cells.shape[0]


def graph_collate(x):
    c, g, v = zip(*x)
    batch_graph = dgl.batch(g)
    batch_values = torch.stack(v, dim=0)
    batch_cells = torch.stack(c, dim=0)
    return batch_cells, batch_graph, batch_values.view(len(g), 1).float()
