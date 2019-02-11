# seq x gene

<!--- 
= TODO 
- describe all the data files in some detail
- details on resuming training
- add detail on evaluation
- cell type as input 
--->

This repository contains code for our paper "Integrating regulatory DNA sequence and gene expression to predict genome-wide chromatin accessibility across cellular contexts". The models are implemented in PyTorch.

## Data

All associated data can be downloaded from [here](http://mitra.stanford.edu/kundaje/projects/seqxgene/).

Untar the `dnase.chr.packbited.tar.gz` file (occupies ~30 Gb).


## Model Training 

### Stage 1

The stage 1 models predict accessibility across all training cell types from only sequence, and does not utilise RNA-seq profiles.

The `model_zoo/stage1` directory contains models for the [Vanilla](./model_zoo/stage1/vanilla.py), [Factorized](./model_zoo/stage1/factorized.py) and our [ResNet](./model_zoo/stage1/resnet.py) models.

To start training any of these models (say, ResNet), from the `model_zoo/stage1` directory:

```bash
python resnet.py -cp /path/to/stage1/checkpoint/dir --dnase /path/to/dnase/packbited --rna_quants /path/to/rna_quants_1630tf.joblib
```

For other inputs, such as hyperparameters, refer

```bash
python resnet.py --help
```

### Stage 2

The stage 2 models predict accessibility for each cell type, sequence pair and uses RNA-seq profiles.

The `model_zoo/stage2` directory contains models for the stage 2 models, which may be trained with or without mean accessibility feature as input (explained in more detail in the paper).

To start training any of these models (say, ResNet, with mean), from the `model_zoo/stage2` directory:

```bash
python simple.py -cp /path/to/stage2/checkpoint/dir --dnase /path/to/dnase/packbited --rna_quants /path/to/rna_quants_1630tf.joblib --stage1_file ../stage1/resnet.py --stage1_pretrained_model_path /path/to/stage1/checkpoint/dir --with_mean 1
```

The model loads weights from the best model from the stage 1 checkpoint directory. You may resume training from a previous checkpoint by adding the argument ```-rb 1``` to the above command. To predict on the test set, add the argument ```-ev 1``` to the above command. This will generate a report of performance on the test set and also produce precision-recall plots. 

For other inputs, such as hyperparameters, refer

```bash
python simple.py --help
```

## Citation

If you use this code for your research, please cite our paper.

<!--- add citation --->
