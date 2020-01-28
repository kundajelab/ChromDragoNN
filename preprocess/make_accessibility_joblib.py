import joblib
import argparse
import gzip
import os
from pyfaidx import Fasta
from tqdm import tqdm
import numpy as np

"""
Input file must be a tab-separated gzipped file formatted as below.

chr    start  end    task1  task2  ...  taskM
chr1   50     1050       0      0           0
chr1   1000   2000       1      0           1
chr2   100    1100       1      0           1

The output is a set of joblib files, one per chromosome which will
store the one hot sequence of each interval along with labels and other
metadata.
"""

BASE_TO_INDEX = {"A":0, "C":1, "G":2, "T":3}

def fetch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="Input tab-separated gzipped file in the specified format")
    parser.add_argument("--genome_fasta", "-g", type=str, help="Path to genome fasta")
    parser.add_argument("--output_dir", "-o", type=str, help="Output directory where chromosome-wise joblib files will be placed")
    args = parser.parse_args()
    return args

def one_hot_packbit(chrm, start, end, genome):
    seq = str(genome[chrm][start:end])
    mask = np.tile([int(x in BASE_TO_INDEX) for x in seq], (4,1)).T
    indices = np.array([BASE_TO_INDEX[x] if x in BASE_TO_INDEX else 0 for x in seq])
    encoding = np.zeros((len(seq), 4)).astype(np.int8)
    encoding[np.arange(indices.size), indices] = 1
    encoding *= mask

    return np.packbits(encoding, axis=1)

def make_joblib(infile, out_dir, genome):
    print("PREPROCESS::: Reading file into memory")
    f = gzip.open(infile)
    d = [x.decode().strip().split('\t') for x in f]
    f.close()

    if not all([len(x)==len(d[0]) for x in d]):
        print("ERROR: Number of cols not constant")
        exit(1)

    tasks = d[0][3:]
    d = d[1:]

    if not all([int(x[2])-int(x[1])==int(d[0][2])-int(d[0][1]) for x in d]):
        print("ERROR: Not all intervals have the same length")
        exit(1)

    chrms = set([x[0] for x in d])
    dat = {c:{'num_cell_types': len(tasks),
              'labels': [],
              'label_metadata': tasks,
              'example_metadata': [],
              'features': []}
            for c in chrms}
    
    print("PREPROCESS::: Processing file")
    for line in tqdm(d):
        chrm = line[0]
        pos = "{}:{}-{}".format(*line[:3])

        dat[chrm]['example_metadata'].append(pos)
        dat[chrm]['labels'].append([np.int8(x) for x in line[3:]])
        dat[chrm]['features'].append(one_hot_packbit(line[0], int(line[1])-1, int(line[2])-1, genome))

    print("PREPROCESS::: Writing output")
    for chrm in chrms:
        dat[chrm]['labels'] = np.packbits(np.array(dat[chrm]['labels']), axis=1)
        dat[chrm]['features'] = np.array(dat[chrm]['features'])
        joblib.dump(dat[chrm], os.path.join(out_dir, "dnase.{}.packbit.joblib".format(chrm)))
    

if __name__ == "__main__":
    args = fetch_args()
    genome = Fasta(args.genome_fasta)
    make_joblib(args.input, args.output_dir, genome)
