import joblib
import argparse
import gzip
import os
import numpy as np

"""
Input file must be a tab-separated (NOT gzipped) file formatted as below. The gene
expression values must be appropriately normalised. In our paper, we use asinh(TPM) 
values for 1630 TFs. Ensure the number and order of the tasks is the same as in the 
accessibility data.

gene    task1   task2  ...  taskM
MEOX1   3.5189  2.8237      3.7542
SOX8    0.0     0.0         1.9623
...
ZNF195  0.0     0.1232      0.0023

The output is a joblib files with the expression matrix and metadata.
"""

def fetch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="Input tab-separated (NOT gzipped) file in the specified format")
    parser.add_argument("--output_prefix", "-o", type=str, help="Prefix for output file")
    args = parser.parse_args()
    return args

def make_joblib(infile, out_prefix):
    print("PREPROCESS::: Reading input")
    f = open(infile)
    d = [x.strip().split('\t') for x in f]
    f.close()

    if not all([len(x)==len(d[0]) for x in d]):
        print("ERROR: Number of cols not constant")
        exit(1)

    tasks = d[0][1:]
    d = d[1:]

    genes = [x[0] for x in d]
    exps = [[float(y) for y in x[1:]] for x in d]

    print("PREPROCESS::: Writing output")
    joblib.dump({
        "rna_quants": np.array(exps).T,
        "genes": genes,
        "tasks": tasks
        }, "{}.joblib".format(out_prefix))
    

if __name__ == "__main__":
    args = fetch_args() 
    make_joblib(args.input, args.output_prefix)
