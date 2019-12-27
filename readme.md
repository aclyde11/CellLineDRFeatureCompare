This code is used to benchmark model performance for high throughput cell line screens.


## Data Folder Requirements 

- ```drugfeats.pkl``` contains mOrdred descritpors without 3D features, that are scaled and imputed
- ```imputer.pkl``` contains a dictionary {imputer, scaler} to be used for on the fly feature generation
- ```rnaseq.pkl``` contains columns with label combo_auc.DRUG, combo_auc.AUC, combo_auc.CELL 
- ```extended_combined_smiles``` which matches smiles to combo_auc.DRUG 
- ```cellpickle.pkl``` contains RNAseq data frame with label lincs.CELL for merging
- ```extended_combined_smiles``` contains smiles matching combo_auc.DRUG
- ```extended_combined_mordred_descriptors``` contains precomputed mordred descriptors for combo_auc.DRUG
- ```testsmiles.smi``` is a SMILES formated file with 100k random sample smiles for testing

## Training Instructions 
Training is done either with --mode [graph, desc, image] (RNN SMILES coming soon). Use ```python train.py -h``` for options.

For this benchmark the following commands were used:
```shell script
python train.py --mode graph -o saved_models/graph_model.pt -w 32 -s cell
python train.py --mode desc -o saved_models/desc_model.pt -w 32 -s cell
python train.py --mode image -o saved_models/image_model.pt -w 32 -s cell
```

## Throughput Benchmarking Instructions
Again use ```python infer.py -h``` to see all options. 

For this benchmark the following commands were used:

```shell script
python infer.py --mode graph -o saved_models/graph_model.pt -w 32 -g 2 --smiles_file data/testsmiles.smi --output_file saved_models/graph_infers.txt
python infer.py --mode desc -o saved_models/desac_model.pt -w 32 -g 2 --smiles_file data/testsmiles.smi --output_file saved_models/desc_infers.txt
python infer.py --mode image -o saved_models/image_model.pt -w 32 -g 2 --smiles_file data/testsmiles.smi --output_file saved_models/image_infers.txt
``` 

## Results