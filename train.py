import argparse

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from features.datasets import VectorDataset, VectorDatasetOnFly, GraphDataset, GraphDatasetOnFly, graph_collate, \
    ImageDatasetOnFly, ImageDataset, SmilesDataset
from features.generateFeatures import smile_to_smile_to_image
from features.utils import get_dgl_graph
from metrics import trackers, rds
from models import basemodel, vectormodel, graphmodel, imagemodel, smilesmodel
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_optimizer(c):
    if c == 'sgd':
        return torch.optim.SGD
    elif c == 'adam':
        return torch.optim.Adam
    elif c == 'adamw':
        return torch.optim.AdamW


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['graph', 'image', 'desc', 'smiles'], required=True,
                        help='model and feature style to use.')
    parser.add_argument('-w', type=int, default=8, help='number of workers for data loaders to use.')
    parser.add_argument('-b', type=int, default=64, help='batch size to use')
    parser.add_argument('-s', choices=['cell', 'drug', 'random', 'hard'], default='cell',
                        help='split style to perform for training')
    parser.add_argument('-o', type=str, default='saved_models/model.pt', help='name of file to save model to')
    parser.add_argument('-r', type=int, default=32, help='random seed for splitting.')
    parser.add_argument('-g', type=int, default=1, help='use data parallel.')
    parser.add_argument('--amp', action='store_true', help='use amp fp16')
    parser.add_argument('--metric_plot_prefix', default=None, type=str, help='prefix for graphs for performance')
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer to use',
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', default=1e-4, type=float, help='learning to use')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs to use')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')

    args = parser.parse_args()
    if args.metric_plot_prefix is None:
        args.metric_plot_prefix = "".join(args.o.split(".")[:-1]) + "_"
    args.optimizer = get_optimizer(args.optimizer)
    return args


def trainer(model, optimizer, train_loader, test_loader, mode, epochs=5):
    tracker = trackers.PytorchHistory()
    lr_red = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, cooldown=0, verbose=True, threshold=1e-4,
                               min_lr=1e-8)

    for epochnum in range(epochs):
        train_loss = 0
        test_loss = 0
        train_iters = 0
        test_iters = 0
        model.train()
        for i, (rnaseq, drugfeats, value) in enumerate(train_loader):
            optimizer.zero_grad()

            if mode == 'desc' or mode == 'image':
                rnaseq, drugfeats, value = rnaseq.to(device), drugfeats.to(device), value.to(device)
                pred = model(rnaseq, drugfeats)
            else:
                rnaseq, value = rnaseq.to(device), value.to(device)
                g = drugfeats
                h = g.ndata['atom_features'].to(device)
                pred = model(rnaseq, g, h)
            mse_loss = torch.nn.functional.mse_loss(pred, value).mean()

            if args.amp:
                with amp.scale_loss(mse_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                mse_loss.backward()
            optimizer.step()
            train_loss += mse_loss.item()
            train_iters += 1
            tracker.track_metric(pred.detach().cpu().numpy(), value.detach().cpu().numpy())

        tracker.log_loss(train_loss / train_iters, train=True)
        tracker.log_metric(internal=True, train=True)

        model.eval()
        with torch.no_grad():
            for i, (rnaseq, drugfeats, value) in enumerate(test_loader):
                if mode == 'desc' or mode == 'image':
                    rnaseq, drugfeats, value = rnaseq.to(device), drugfeats.to(device), value.to(device)
                    pred = model(rnaseq, drugfeats)
                else:
                    rnaseq, value = rnaseq.to(device), value.to(device)
                    g = drugfeats
                    h = g.ndata['atom_features'].to(device)
                    pred = model(rnaseq, g, h)
                mse_loss = torch.nn.functional.mse_loss(pred, value).mean()
                test_loss += mse_loss.item()
                test_iters += 1
                tracker.track_metric(pred.detach().cpu().numpy(), value.detach().cpu().numpy())
        tracker.log_loss(train_loss / train_iters, train=False)
        tracker.log_metric(internal=True, train=False)

        lr_red.step(test_loss / test_iters)
        print("Epoch", epochnum, train_loss / train_iters, test_loss / test_iters, 'r2',
              tracker.get_last_metric(train=True), tracker.get_last_metric(train=False))

    if args.g == 1:
        torch.save({'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'inference_model': model,
                    'history': tracker}, args.o)
    else :
        torch.save({'model_state': model.module.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'inference_model': model.module,
                    'history': tracker}, args.o)
    return model, tracker


def produce_preds_timing(model, loader, cells, drugs, mode):
    preds = []
    values = []
    model.eval()
    with torch.no_grad():
        for (rnaseq, drugfeats, value) in loader:
            if mode == 'desc' or mode == 'image' or mode == 'smiles':
                rnaseq, drugfeats, value = rnaseq.to(device), drugfeats.to(device), value.to(device)
                pred = model(rnaseq, drugfeats)
            else:
                rnaseq, value = rnaseq.to(device), value.to(device)
                g = drugfeats
                h = g.ndata['atom_features'].to(device)
                pred = model(rnaseq, g, h)
            preds.append(pred.cpu().detach().numpy())
            values.append(value.cpu().detach().numpy())

    preds = np.concatenate(preds).flatten()
    values = np.concatenate(values).flatten()
    res = np.stack([preds, values, cells, drugs])
    np.save(args.o + ".npy", res)
    return res


if __name__ == '__main__':
    args = get_args()

    np.random.seed(args.r)
    torch.manual_seed(args.r)

    dataloader = None
    model = None

    print("Loading base frame. ")
    cell_frame = pd.read_pickle("data/cellpickle.pkl")
    base_frame = pd.read_pickle("data/rnaseq.pkl")
    smiles_frame = pd.read_csv("data/extended_combined_smiles")
    good_drugs = []
    for i in tqdm(range(smiles_frame.shape[0])):
        smi = smiles_frame.iloc[i, 1]
        test = Chem.MolFromSmiles(smi)
        if test is not None:
            good_drugs.append(smiles_frame.iloc[i, 0])

    base_frame = base_frame[base_frame['auc_combo.DRUG'].isin(good_drugs)]
    print("Done, base frame is shape", base_frame.shape)

    if args.s == 'cell':
        print("Splitting on cells...")
        train_idx, test_idx = train_test_split(list(range(base_frame.shape[0])), stratify=base_frame['auc_combo.CELL'],
                                               test_size=0.2, random_state=args.r)
    elif args.s == 'drug':
        print("Splitting on drugs...")
        train_idx, test_idx = train_test_split(list(range(base_frame.shape[0])), stratify=base_frame['auc_combo.DRUG'],
                                               test_size=0.2, random_state=args.r)
    elif args.s == 'hard':
        unique_cells = np.unique(np.array(base_frame['auc_combo.CELL']))
        unique_drugs = np.unique(np.array(base_frame['auc_combo.DRUG']))

        train_drugs, _ = map(list, train_test_split(unique_cells, test_size=0.2, random_state=args.r))
        train_cells, _ = map(list, train_test_split(unique_drugs, test_size=0.2, random_state=args.r))

        train_idx = []
        test_idx = []
        for i, (index, row) in enumerate(base_frame.iterrows()):
            if row['auc_combo.CELL'] in train_cells or row['auc_combo.DRUG'] in train_drugs:
                train_idx.append(i)
            else:
                test_idx.append(i)
        train_idx, test_idx = map(np.array, [train_idx, test_idx])
    else:
        print("Splitting randomly...")
        train_idx, test_idx = train_test_split(list(range(base_frame.shape[0])),
                                               test_size=0.2, random_state=args.r)

    cells = np.array(base_frame['auc_combo.CELL'])
    values = np.array(base_frame['auc_combo.AUC'], dtype=np.float32)[np.newaxis, :]
    drugs = np.array(base_frame['auc_combo.DRUG'])
    print("Done loading and splitting base frames...")

    smiles_frame = pd.read_csv("data/extended_combined_smiles")
    smiles_frame = smiles_frame.set_index('NAME')
    smiles = {}
    for index, row in tqdm(smiles_frame.iterrows()):
        smiles[index] = (row.iloc[0])

    if args.mode == 'graph':
        frame = {}

        print("Producing graph features...")
        for index, row in tqdm(smiles_frame.iterrows()):
            try:
                frame[index] = get_dgl_graph(row['SMILES'])
            except AttributeError:
                continue

        train_dset = GraphDataset(cells[train_idx], cell_frame, frame, values[:, train_idx], drugs[train_idx])
        test_dset = GraphDataset(cells[test_idx], cell_frame, frame, values[:, test_idx], drugs[test_idx])

        train_loader = DataLoader(train_dset, collate_fn=graph_collate, shuffle=True, num_workers=args.w,
                                  batch_size=args.b)
        test_loader = DataLoader(test_dset, collate_fn=graph_collate, shuffle=True, num_workers=args.w,
                                 batch_size=args.b)

        test_dset = GraphDatasetOnFly(cells[test_idx], cell_frame, smiles, values[:, test_idx], drugs[test_idx])
        # test_loader_fly = DataLoader(test_dset, collate_fn=graph_collate, shuffle=False, num_workers=args.w,
        #                              batch_size=args.b)

        model = basemodel.BaseModel(cell_frame.shape[1] - 1, args.dropout_rate, featureModel=graphmodel.GCN,
                                    intermediate_rep_drugs=128,
                                    flen=frame[list(frame.keys())[0]].ndata['atom_features'].shape[1])


    elif args.mode == 'desc':
        desc_data_frame = pd.read_pickle("data/drugfeats.pkl")
        desc_data_frame = desc_data_frame.set_index("DRUG")
        frame = {}

        print("Producing desc features...")
        for ind in range(desc_data_frame.shape[0]):
            frame[desc_data_frame.index[ind]] = np.array(desc_data_frame.iloc[ind], dtype=np.float32)

        train_dset = VectorDataset(cells[train_idx], cell_frame, frame, values[:, train_idx], drugs[train_idx])
        test_dset = VectorDataset(cells[test_idx], cell_frame, frame, values[:, test_idx], drugs[test_idx])

        train_loader = DataLoader(train_dset, shuffle=True, num_workers=args.w, batch_size=args.b)
        test_loader = DataLoader(test_dset, shuffle=True, num_workers=args.w, batch_size=args.b)

        model = basemodel.BaseModel(cell_frame.shape[1] - 1, args.dropout_rate, featureModel=vectormodel.VectorModel,
                                    intermediate_rep_drugs=128, flen=desc_data_frame.shape[1])

    elif args.mode == 'image':
        frame = {}

        print("Producing image features.")
        for index, row in tqdm(smiles_frame.iterrows()):
            try:
                frame[index] = smile_to_smile_to_image(row['SMILES'])
            except AttributeError:
                continue

        train_dset = ImageDataset(cells[train_idx], cell_frame, frame, values[:, train_idx], drugs[train_idx])
        test_dset = ImageDataset(cells[test_idx], cell_frame, frame, values[:, test_idx], drugs[test_idx])

        train_loader = DataLoader(train_dset, shuffle=True, num_workers=args.w, batch_size=args.b)
        test_loader = DataLoader(test_dset, shuffle=True, num_workers=args.w, batch_size=args.b)

        model = basemodel.BaseModel(cell_frame.shape[1] - 1, args.dropout_rate, featureModel=imagemodel.ImageModel,
                                    intermediate_rep_drugs=128, flen=None)
    elif args.mode == 'smiles':
        frame = {}

        print("Producing smile features.")
        for index, row in tqdm(smiles_frame.iterrows()):
            frame[index] = row['SMILES']

        train_dset = SmilesDataset(cells[train_idx], cell_frame, frame, values[:, train_idx], drugs[train_idx])
        test_dset = SmilesDataset(cells[test_idx], cell_frame, frame, values[:, test_idx], drugs[test_idx])

        train_loader = DataLoader(train_dset, shuffle=True, num_workers=args.w, batch_size=args.b)
        test_loader = DataLoader(test_dset, shuffle=True, num_workers=args.w, batch_size=args.b)


        model = basemodel.BaseModel(cell_frame.shape[1] - 1, args.dropout_rate, featureModel=smilesmodel.SmilesModel,
                                    intermediate_rep_drugs=128, flen=None)

    print("Done.")

    print("Starting trainer.")
    if args.g > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
        optimizer = args.optimizer(model.parameters(), lr=args.lr)

    elif args.amp:
        model.to(device)
        optimizer = args.optimizer(model.parameters(), lr=args.lr)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        model.to(device)
        optimizer = args.optimizer(model.parameters(), lr=args.lr)

    print("Number of parameters:",
          sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    model, history = trainer(model, optimizer, train_loader, test_loader, mode=args.mode, epochs=args.epochs)
    history.plot_loss(save_file=args.metric_plot_prefix + "loss.png", title=args.mode + " Loss")
    history.plot_metric(save_file=args.metric_plot_prefix + "r2.png", title=args.mode + " " + history.metric_name)
    print("Finished training, now")

    print("Running evaluation for surface plots...")
    res = produce_preds_timing(model, test_loader, cells[test_idx], drugs[test_idx], args.mode)
    rds_model = rds.RegressionDetectionSurface(percent_min=-3)
    rds_model.compute(res[1], res[0], stratify=res[2], samples=30)
    rds_model.plot(save_file=args.metric_plot_prefix + "rds_on_cell.png",
                   title='Regression Enrichment Surface (Avg over Unique Cells)')
    rds_model.compute(res[1], res[0], stratify=res[3], samples=30)
    rds_model.plot(save_file=args.metric_plot_prefix + "rds_on_drug.png",
                   title='Regression Enrichment Surface (Avg over Unique Drugs)')
    print("Output all plots with prefix", args.metric_plot_prefix)
