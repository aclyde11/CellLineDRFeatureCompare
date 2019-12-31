import io

import cairosvg
import numpy as np
from PIL import Image
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torchvision.transforms import ToTensor
import dgl
from features import utils as featmaker
from features.smiles import smi_tokenizer
from features.utils import Invert

smiles_vocab = None  # load charater to int function
data_location = 'data/'


def smile_to_mordred(smi, imputer_dict=None):
    smi = Chem.MolFromSmiles(smi)
    calc = Calculator(descriptors, ignore_3D=True)
    res = calc(smi)
    res = np.array(list(res.values())).reshape(1, -1)
    if imputer_dict is not None:
        imputer_dict = imputer_dict[0]
        res = imputer_dict['scaler'].transform(imputer_dict['imputer'].transform(res))
    return res.flatten().astype(np.float32)


def smile_to_smile_to_image(mol, molSize=(128, 128), kekulize=True, mol_name=''):
    mol = Chem.MolFromSmiles(mol)
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    image = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg, parent_width=100, parent_height=100,
                                                   scale=1)))
    image.convert('RGB')
    return ToTensor()(Invert()(image)).numpy()


def smiles_to_smiles(smi, vocab, maxlen=320):
    t = [vocab[i] for i in smi_tokenizer(smi)]
    if len(t) >= maxlen:
        t = t[:maxlen]
    else:
        t = t + (maxlen - len(t)) * [vocab[' ']]
    t = np.array(t).flatten()
    return t


def smiles_to_graph(mol, args):
    return featmaker.get_dgl_graph(mol)

def smiles_to_graph_batch(mol, args, batch=True):
    gs = featmaker.get_dgl_graph_batch(mol, args)
    if batch:
        return dgl.batch(gs)
    else:
        return gs