"""Description: preprocessing functions that take standard
bioinformatic formats (BED, FASTA, etc) and format into hdf5
files that are input to deep learning models
"""

import os
import gzip
import glob
import h5py
import time
import logging

import numpy as np

from bed import bin_regions_parallel
from bed import split_bed_to_chrom_bed_parallel
from bed import setup_negatives
from bed import generate_labels
from metadata import save_metadata
from parallelize import setup_multiprocessing_queue
from parallelize import run_in_parallel


_CHROM_TAG = "chromosomes"
_EXAMPLE_TYPE_TAG = "example_type"


def setup_h5_dataset(
        bin_file,
        ref_fasta,
        chromsizes,
        h5_file,
        label_sets,
        signal_sets,
        bin_size,
        stride,
        final_length,
        reverse_complemented,
        onehot_features_key,
        tmp_dir,
        binned=True):
    """given a region file, set up dataset
    conventionally, this is 1 chromosome
    """
    # make sure tmp dir exists
    os.system("mkdir -p {}".format(tmp_dir))
    
    # set up prefix
    prefix = os.path.basename(bin_file).split(
        ".narrrowPeak")[0].split(".bed")[0]

    # save in the metadata
    save_metadata(bin_file, h5_file)
    
    # generate BED annotations on the active center
    for key in label_sets.keys():
        label_files = label_sets[key][0]
        label_params = label_sets[key][1]
        method = label_params.get("method", "half_peak")
        generate_labels(
            bin_file, label_files, key, h5_file,
            method=method, chromsizes=chromsizes,
            tmp_dir=tmp_dir)
    
    return h5_file


def generate_h5_datasets(
        positives_bed_file,
        ref_fasta,
        chromsizes,
        label_files,
        signal_files,
        prefix,
        work_dir,
        bin_size=200,
        stride=50,
        final_length=1000,
        superset_bed_file=None,
        reverse_complemented=False,
        genome_wide=False,
        parallel=24,
        tmp_dir=".",
        normalize_signals=False):
    """generate a full h5 dataset
    """
    if True:
        # first select negatives
        training_negatives_bed_file, genomewide_negatives_bed_file = setup_negatives(
            positives_bed_file,
            superset_bed_file,
            chromsizes,
            bin_size=bin_size,
            stride=stride,
            genome_wide=genome_wide,
            tmp_dir=tmp_dir)

        # collect the bed files
        if genome_wide:
            all_bed_files = [
                positives_bed_file,
                training_negatives_bed_file,
                genomewide_negatives_bed_file]
        else:
            all_bed_files = [
                positives_bed_file,
                training_negatives_bed_file]

        # split to chromosomes
        chrom_dir = "{}/by_chrom".format(tmp_dir)
        os.system("mkdir -p {}".format(chrom_dir))
        split_bed_to_chrom_bed_parallel(
            all_bed_files, chrom_dir, parallel=parallel)

        # split to equally sized bin groups
        chrom_files = glob.glob("{}/*.bed.gz".format(chrom_dir))
        bin_dir = "{}/bin-{}.stride-{}".format(tmp_dir, bin_size, stride)
        os.system("mkdir -p {}".format(bin_dir))
        bin_regions_parallel(
            chrom_files, bin_dir, chromsizes, bin_size=bin_size, stride=stride, parallel=parallel)

        # grab all of these and process in parallel
        h5_dir = "{}/h5".format(work_dir)
        os.system("mkdir -p {}".format(h5_dir))
        chrom_bed_files = glob.glob("{}/*.filt.bed.gz".format(bin_dir))
        logging.info("Found {} bed files".format(chrom_bed_files))
        h5_queue = setup_multiprocessing_queue()
        for bed_file in chrom_bed_files:
            prefix = os.path.basename(bed_file).split(".bed")[0].split(".narrowPeak")[0]
            h5_file = "{}/{}.h5".format(h5_dir, prefix)
            if os.path.isfile(h5_file):
                continue
            parallel_tmp_dir = "{}/{}_tmp".format(tmp_dir, prefix)
            process_args = [
                bed_file,
                ref_fasta,
                chromsizes,
                h5_file,
                label_files,
                signal_files,
                bin_size,
                stride,
                final_length,
                reverse_complemented,
                "features",
                parallel_tmp_dir]
            h5_queue.put([setup_h5_dataset, process_args])

        # run the queue
        run_in_parallel(h5_queue, parallel=parallel, wait=True)

    # also tag each file with the chromosome and positives, negatives, etc
    h5_dir = "{}/h5".format(work_dir)
    h5_files = glob.glob("{}/*h5".format(h5_dir))
    for h5_file in h5_files:
        chrom = os.path.basename(h5_file).split(".")[-4]
        example_type = os.path.basename(h5_file).split(".")[-5]
        if example_type == "master":
            example_type = "positives"
        with h5py.File(h5_file, "a") as hf:
            hf["/"].attrs[_CHROM_TAG] = [chrom]
            hf["/"].attrs[_EXAMPLE_TYPE_TAG] = example_type
    
    return None
