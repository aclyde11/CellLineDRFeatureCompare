import argparse
import multiprocessing
import pickle
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from features.generateFeatures import smiles_to_graph, smile_to_smile_to_image, smile_to_mordred

if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['graph', 'image', 'desc'], required=True,
                        help='model and feature style to use.')
    parser.add_argument('-w', type=int, default=8, help='number of workers for data loaders to use.')
    parser.add_argument('-g', type=int, default=1, help='number of gpu workers for inference to use.')
    parser.add_argument('-b', type=int, default=64, help='batch size to use')

    parser.add_argument('-o', type=str, default='saved_models/model.pt', help='name of file to save model to')
    parser.add_argument('-r', type=int, default=32, help='random seed for splitting.')

    parser.add_argument('--num_smiles', type=int, required=False, help='Limit number of smiles for testing.')
    parser.add_argument('--smiles_file', type=str, required=True, help='SMILES format file to use for inferences.')
    parser.add_argument('--output_file', type=str, required=True, help='Output for predictions.')
    return parser.parse_args()


def get_feature_prodcer(mode):
    if mode == 'desc':
        with open("data/imputer.pkl", 'rb') as f:
            imps = pickle.load(f)
        args = (imps)
        return smile_to_mordred, args
    elif mode == 'graph':
        return smiles_to_graph
    elif mode == 'image':
        return smile_to_smile_to_image


'''
    cell_features is a numpy array of feature data without names
    cell_names is a matching list of cell names. 
'''


def feature_worker(args, smile_queue, feature_queue, cell_features, cell_names, id, stop):
    iter_counter = 0
    feature_producer, argsfp = get_feature_prodcer(args.mode)

    while not stop.value or not smiles_queue.empty():
        while not smile_queue.empty():
            res = smile_queue.get(timeout=10)
            if res is not None:
                smile, drug_name = res
                try:
                    drug_features = feature_producer(smile, argsfp)
                    assert (drug_features is not None)
                except AssertionError:
                    print("Smile error....")
                    continue
                feature_queue.put(
                    (torch.from_numpy(drug_features).float().unsqueeze(0).repeat([cell_features.shape[0], 1]),
                     torch.from_numpy(cell_features).float(),
                     smile, drug_name, cell_names))
            iter_counter += 1
            if iter_counter % 100 == 0:
                print(id, "did ", iter_counter)
        time.sleep(3)


def infer(feature_queue, out_queue, model_path, cuda_id, mode, smiles_counter, stop):
    print("Starting gpu worker", cuda_id)
    iter_counter = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(cuda_id))
    else:
        device = torch.device("cpu")

    model = torch.load(model_path, map_location=device)
    model = model['inference_model']
    model.eval()
    print("Model loaded.")

    with torch.no_grad():
        while not stop.value:
            while not feature_queue.empty():
                res = feature_queue.get(timeout=10)
                if res is not None:
                    smiles_counter.value += 1
                    drug_features, cell_features, smile, name, cell_names = res
                    if mode == 'desc' or mode == 'image':
                        drug_features = drug_features.to(device)
                        cell_features = cell_features.to(device)
                        preds = model(cell_features, drug_features)
                    else:
                        rnaseq, value = rnaseq.to(device), value.to(device)
                        g = drug_features
                        h = g.ndata['atom_features'].to(device)
                        preds = model(rnaseq, g, h)
                    preds = preds.detach().cpu().numpy().flatten()
                    out_queue.put({'preds': preds, 'smile': smile, 'drug_name': name, 'cell_names': cell_names})
                iter_counter += 1
                if iter_counter % 100 == 0:
                    print('cuda', cuda_id, "did ", iter_counter)
            time.sleep(3)

    return True


def writer_worker(outfile, out_queue, stop):
    with open(outfile, 'w') as f:
        f.write(",".join(["cell_name", "drug_name", 'drug_smiles', "pred_auc"]) + "\n")
        while not stop.value:
            while not out_queue.empty():
                res = out_queue.get(timeout=10)

                if res is not None:
                    preds = res['preds']
                    drug_name = res['drug_name']
                    smile = res['smile']
                    for i, cell_name in enumerate(res['cell_names']):
                        f.write(
                            ",".join([str(cell_name), str(drug_name), str(smile), str(preds[i])]) + '\n'
                        )
    return True


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")  # needed for cuda.
    args = get_args()

    np.random.seed(args.r)
    torch.manual_seed(args.r)  # may not be fully reproducible without deterministic cuda, but this should be ok.

    print("Loading base frame. ")
    cell_frame = pd.read_pickle("data/cellpickle.pkl")
    cell_names = cell_frame.iloc[:, 0]
    cell_features = np.array(cell_frame.iloc[:, 1:], dtype=np.float32)
    print("Extracted", cell_frame.shape[0], 'cells for inference.')

    ## Loading smiles frame.
    smiles = pd.read_csv(args.smiles_file, sep=' ', header=None, names=['SMILES', 'name'])
    if args.num_smiles is not None:
        args.num_smiles = min(args.num_smiles, smiles.shape[0])
        print("Limiting smiles to", args.num_smiles)
        smiles = smiles.iloc[:args.num_smiles]
    print("Loaded", smiles.shape[0], "SMILES.")

    feature_workers = []
    gpu_workers = []
    smiles_queue = multiprocessing.Queue()
    feature_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()

    smiles_counter = multiprocessing.Value('i', 0)
    workers_stop = multiprocessing.Value('b', False)
    writer_stop = multiprocessing.Value('b', False)
    gpu_stop = multiprocessing.Value('b', False)

    for i in range(args.w):
        feature_workers.append(multiprocessing.Process(target=feature_worker, args=(
            args, smiles_queue, feature_queue, cell_features, cell_names, i, workers_stop)))
    for i in range(args.g):
        gpu_workers.append(
            multiprocessing.Process(target=infer,
                                    args=(feature_queue, out_queue, args.o, i, args.mode, smiles_counter, gpu_stop)))
    writer_proc = multiprocessing.Process(target=writer_worker, args=(args.output_file, out_queue, writer_stop))

    writer_proc.start()
    for proc in gpu_workers:
        proc.start()
    for proc in feature_workers:
        proc.start()

    start_time = time.time()

    print("Putting smiles in queue...")
    for i, row in tqdm(smiles.iterrows(), desc='smile queue filling'):
        smiles_queue.put((row['SMILES'], row['name']))

    while not smiles_queue.empty():
        try:
            print("SMILES QUEUE", smiles_queue.qsize(),
                  "FEATURE QUEUE", feature_queue.qsize(),
                  "OUT QUEUE", out_queue.qsize())
        except NotImplementedError:
            print("Queue size not avilable on your system...\n",
                  "SMILES QUEUE empty", smiles_queue.empty(),
                  "FEATURE QUEUE empty", feature_queue.empty(),
                  "OUT QUEUE empty", out_queue.empty()
                  )
        time.sleep(10)
    workers_stop.value = True
    print("Turning off feature workers.")

    while not feature_queue.empty():
        try:
            print("SMILES QUEUE", smiles_queue.qsize(),
                  "FEATURE QUEUE", feature_queue.qsize(),
                  "OUT QUEUE", out_queue.qsize())
        except NotImplementedError:
            print("Queue size not avilable on your system...\n",
                  "SMILES QUEUE empty", smiles_queue.empty(),
                  "FEATURE QUEUE empty", feature_queue.empty(),
                  "OUT QUEUE empty", out_queue.empty()
                  )
        time.sleep(10)
    print("Turning off gpu worker.")
    gpu_stop.value = True
    end_time = time.time()

    while not out_queue.empty():
        try:
            print("SMILES QUEUE", smiles_queue.qsize(),
                  "FEATURE QUEUE", feature_queue.qsize(),
                  "OUT QUEUE", out_queue.qsize())
        except NotImplementedError:
            print("Queue size not avilable on your system...\n",
                  "SMILES QUEUE empty", smiles_queue.empty(),
                  "FEATURE QUEUE empty", feature_queue.empty(),
                  "OUT QUEUE empty", out_queue.empty()
                  )
        time.sleep(10)
    print("Turning off writer worker.")
    writer_stop.value = True

    for proc in feature_workers:
        proc.join()
    writer_proc.join()
    for proc in gpu_workers:
        proc.join()
    print("Done.")

    print("total smiles inferenced on: ", smiles_counter.value)
    print("total time", end_time - start_time)
    print("Smiles per second", smiles_counter.value / (end_time - start_time))
