# ChromDragoNN: cis-trans Deep RegulAtory Genomic Neural Network for predicting Chromatin Accessibility

<!--- 
= TODO 
- describe all the data files in some detail
- details on resuming training
- add detail on evaluation
- cell type as input 
--->

This repository contains code for our paper "Integrating regulatory DNA sequence and gene expression to predict genome-wide chromatin accessibility across cellular contexts". The models are implemented in PyTorch.

## Data

All associated data from our paper can be downloaded from [here](http://mitra.stanford.edu/kundaje/projects/seqxgene/) or [here](https://zenodo.org/record/2635744#.XjCuIC_MxTY).

Untar the `dnase.chr.packbited.tar.gz` file (occupies ~30 Gb).

If you have your own data, you may use scripts in the `preprocess/` directory. 

### Preparing Accessibility Data
For the accessibility matrix, prepare your data in the following format as a tab-separated gzipped file. 
```
chr    start  end    task1  task2  ...  taskM
chr1   50     1050       0      0           0
chr1   1000   2000       1      0           1
...
chr2   100    1100       1      0           1
```

ChromDragoNN works on binary data and hence do ensure that the labels are all 0 or 1 only.

Then use the following command to process the data (this may take a few hours depending on the size of your dataset):
```bash
python ./preprocess/make_accessibility_joblib.py --input /path/to/accessibility/file.tsv.gz --output_dir /path/to/dnase/packbited --genome_fasta /path/to/genome/fasta.fa
``` 
Make sure the output directory exists!

If you wish to generate the binary matrix from peaks (e.g. narrowPeak), have a look at the [seqdataloader](https://github.com/kundajelab/seqdataloader) repo. 

### Preparing RNA Data
For the RNA matrix, prepare your data in the following format as a tab-separated file (NOT gzipped). 
```
gene    task1   task2  ...  taskM
MEOX1   3.5189  2.8237      3.7542
SOX8    0.0     0.0         1.9623
...
ZNF195  0.0     0.1232      0.0023
```
The gene expression values must already be appropriately normalised. In our paper, we use the arcsinh(TPM) values for 1630 Transcription Factors. Do ensure the number and order of the tasks is the same as in the accessibility data.

Then use the following command to process the data:
```bash
python ./preprocess/make_rna_joblib.py --input /path/to/rna/file.tsv --output_prefix /path/to/rna/prefix
```

This will output `/path/to/rna/prefix.joblib` RNA quants file.


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

The model loads weights from the best model from the stage 1 checkpoint directory. You may resume training from a previous checkpoint by adding the argument ```-rb 1``` to the above command. To predict on the test set, add the arguments ```-rb 1 -ev 1``` to the above command. This will generate a report of performance on the test set and also produce precision-recall plots. 

For other inputs, such as hyperparameters, refer

```bash
python simple.py --help
```

## Citation

If you use this code for your research, please cite our paper:

Surag Nair, Daniel S Kim, Jacob Perricone, Anshul Kundaje, Integrating regulatory DNA sequence and gene expression to predict genome-wide chromatin accessibility across cellular contexts, Bioinformatics, Volume 35, Issue 14, July 2019, Pages i108–i116, https://doi.org/10.1093/bioinformatics/btz352

<!--- add citation --->
