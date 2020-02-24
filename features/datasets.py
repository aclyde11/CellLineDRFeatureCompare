import pickle
import random

import dgl
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from features.generateFeatures import smile_to_mordred, smile_to_smile_to_image, smiles_to_graph
from features.smiles import get_vocab, smi_tokenizer


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
        return torch.from_numpy(cell_data).float(), torch.from_numpy(t).float(), torch.from_numpy(
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
        return torch.from_numpy(cell_data).float(), torch.from_numpy(
            self.graphs[self.drugs[item]]).float(), torch.from_numpy(
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


def randomSmiles_(m1):
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)


def randomSmiles(smi, max_len=150, attempts=100):
    m1 = Chem.MolFromSmiles(smi)
    if m1 is None:
        return None
    if m1 is not None and attempts == 1:
        return [smi]

    s = set()
    for i in range(attempts):
        smiles = randomSmiles_(m1)
        s.add(smiles)
    # s = list(filter(lambda x : len(x) < max_len, list(s)))

    if len(s) > 1:
        return list(s)
    else:
        return [smi]


class SmilesDataset(Dataset):
    def __init__(self, cells, rnaseq, g, values, drugs, random_permutes=True, maxlen=320):
        self.graphs = g
        self.rnaseq = rnaseq
        self.values = values
        self.cells = cells
        self.drugs = drugs
        self.vocab = get_vocab()
        self.random_permutes = random_permutes
        self.maxlen = maxlen

    def __getitem__(self, item):
        t = self.graphs[self.drugs[item]]

        if self.random_permutes:
            try:
                t = Chem.MolFromSmiles(t)
                assert (t is not None)
                t = randomSmiles_(t)
            except:
                pass
        t = t.strip()
        t = [self.vocab[i] for i in smi_tokenizer(t)]
        if len(t) >= self.maxlen:
            t = t[:self.maxlen]
        else:
            t = t + (self.maxlen - len(t)) * [self.vocab[' ']]
        t = np.array(t).flatten()

        cell_data = np.array(self.rnaseq[self.rnaseq['lincs.Sample'] == self.cells[item]].iloc[0, 1:], dtype=np.float32)
        return torch.from_numpy(cell_data).float(), torch.from_numpy(t).long(), torch.from_numpy(
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

class DescGraphDataset(Dataset):
    def __init__(self, cells, rnaseq, g, values, drugs):
        self.graphs, self.descs = g
        self.cells = cells
        self.rnaseq = rnaseq
        self.values = values
        self.drugs = drugs

        self.random_filter_descrip = 0.2

    def __getitem__(self, item):
        gate1 = torch.from_numpy(np.array([1])).float()

        if random.random() < self.random_filter_descrip:
            gate2 = torch.from_numpy(np.array([0])).float()
            t = self.descs[self.drugs[item]]
            t = torch.zeros(t.shape).float()
        else:
            t = torch.from_numpy(self.descs[self.drugs[item]]).float()
            gate2 = torch.from_numpy(np.array([1])).float()

        cell_data = np.array(self.rnaseq[self.rnaseq['lincs.Sample'] == self.cells[item]].iloc[0, 1:], dtype=np.float32)
        return torch.from_numpy(cell_data), gate1, gate2, self.graphs[self.drugs[item]], t, torch.from_numpy(self.values[:, item])

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



def graph_collate(x):
    c, g1, g2, g, d, v = zip(*x)
    batch_graph = dgl.batch(g)
    batch_g1 = torch.stack(g1, dim=0)
    batch_g2 = torch.stack(g2, dim=0)
    batch_desc = torch.stack(d, dim=0)
    batch_values = torch.stack(v, dim=0)
    batch_cells = torch.stack(c, dim=0)
    return batch_cells, (batch_g1, batch_g2, batch_graph, batch_desc), batch_values.view(len(g), 1).float()
