# seq x gene

[TODO]: <>
[describe all the data files in some detail]: <>
[add detail on evaluation]: <>

This repository contains code for our paper "Integrating regulatory DNA sequence and gene expression to predict genome-wide chromatin accessibility across cellular contexts".

### Data

All associated data can be downloaded from [here](http://mitra.stanford.edu/kundaje/projects/seqxgene/).

Untar the `dnase.chr.packbited.tar.gz` file (occupies ~30 Gb).


### Model Training 

#### Stage 1

The stage 1 models predict accessibility across all training cell types from only sequence, and does not utilise RNA-seq profiles.

The `model_zoo/stage1` directory contains models for the [Vanilla](./model_zoo/stage1/basset_vanilla.py), [Factorized]([vanilla](./model_zoo/stage1/basset_factorized.py) and our [ResNet]([vanilla](./model_zoo/stage1/basset_resnet.py) models.

To start training any of these models (say, ResNet), from the `model_zoo/stage1` directory:

`python basset_resnet.py -cp /path/to/stage1/checkpoint/dir --dnase /path/to/dnase/packbited --rna_quants /path/to/rna_quants_1630tf.joblib`

For other inputs, such as hyperparameters, refer

`python basset_resnet.py --help`

#### Stage 2

The stage 2 models predict accessibility for each cell type, sequence pair and uses RNA-seq profiles.

The `model_zoo/stage2` directory contains models for the respective stage 1 models, with 2 variants each- with and without mean accessibility feature as input (explained in more detail in the paper).

To start training any of these models (say, ResNet, with mean), from the `model_zoo/stage2/resnet` directory:

`python resnet_mean.py -cp /path/to/stage2/checkpoint/dir --dnase /path/to/dnase/packbited --rna_quants /path/to/rna_quants_1630tf.joblib --basset_pretrained_path /path/to/stage1/checkpoint/dir`

The model loads weights from the best model from the stage 1 checkpoint directory. For other inputs, such as hyperparameters, refer

`python basset_resnet.py --help`

### Citation

If you use this code for your research, please cite our paper:
