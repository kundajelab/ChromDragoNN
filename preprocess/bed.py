# descriptions: preprocess information from BED files

import os
import h5py
import gzip
import glob
import logging

import numpy as np
import pandas as pd

from h5_utils import AttrKeys
from parallelize import setup_multiprocessing_queue
from parallelize import run_in_parallel
from utils import MetaKeys


def generate_master_regions(master_bed_file, bed_files):
    """Generate master regions file
    
    Args:
      master_bed_file: BED file for master regions set
      bed_files: BED files to combine as master region set

    Returns:
      None
    """
    logging.info("generate master regions file")
    assert len(bed_files) > 0
    tmp_master_bed_file = '{}.tmp.bed.gz'.format(
        master_bed_file.split('.bed')[0].split('.narrowPeak')[0])
    
    for i in xrange(len(bed_files)):
        bed_file = bed_files[i]
        if i == 0:
            # for first file copy into master
            transfer = "cp {0} {1}".format(bed_file, master_bed_file)
            logging.debug(transfer)
            os.system(transfer)
        else:
            # merge master with current bed
            merge_bed = (
                "zcat {0} {1} | "
                "awk -F '\t' '{{ print $1\"\t\"$2\"\t\"$3 }}' | "
                "sort -k1,1 -k2,2n | "
                "bedtools merge -i - | "
                "gzip -c > {2}").format(
                    bed_file,
                    master_bed_file,
                    tmp_master_bed_file)
            logging.debug(merge_bed)
            os.system(merge_bed)

            # copy tmp master over to master
            transfer = "cp {0} {1}".format(tmp_master_bed_file, master_bed_file)
            logging.debug(transfer)
            os.system(transfer)
                
    os.system('rm {}'.format(tmp_master_bed_file))

    return None


def split_bed_to_chrom_bed(
        out_dir,
        peak_file,
        prefix):
    """Splits a gzipped peak file into its various chromosomes

    Args:
      out_dir: output directory to put chrom files
      peak_file: BED file of form chr, start, stop (tab delim)
                 Does not need to be sorted
      prefix: desired prefix on chrom files

    Returns:
      None
    """
    logging.info("splitting BED file into chromosome BED files...")
    assert os.path.splitext(peak_file)[1] == '.gz'

    current_chrom = ''
    #data_dict = {}
    with gzip.open(peak_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')
            chrom = fields[0]
            chrom_file = '{0}/{1}.{2}.bed.gz'.format(out_dir, prefix, chrom)
            #data_dict[chrom] = {"bed": chrom_file}
            with gzip.open(chrom_file, 'a') as out_fp:
                out_fp.write(line)
            
            # Tracking
            if chrom != current_chrom:
                logging.info("Started {}".format(chrom))
                current_chrom = chrom

    return None


def split_bed_to_chrom_bed_parallel(
        bed_files,
        out_dir,
        parallel=12):
    """split a list of bed files into chromosome bed files, in parallel
    """
    # put commands in queue
    split_queue = setup_multiprocessing_queue()
    for bed_file in bed_files:
        prefix = os.path.basename(bed_file).split(".narrowPeak")[0].split(".bed")[0]
        split_args = [
            out_dir,
            bed_file,
            prefix]
        split_queue.put([split_bed_to_chrom_bed, split_args])

    # run the queue
    run_in_parallel(split_queue, parallel=parallel, wait=True)

    return None


def _get_adjusted_start_and_stop(
        start, stop, method, num_flanks=3):
    """adjust the start and stop of a region based on whether
    flanks should be present
    """
    if method == 'naive':
        # Just go from start of region to end of region
        mark = start
        adjusted_stop = stop
    elif method == 'plus_flank_negs':
        # Add 3 flanks to either side
        mark = max(start - num_flanks * stride, 0)
        adjusted_stop = stop + num_flanks * stride
    # add other binning strategies as needed here
    else:
        raise ValueError, "method not yet implemented!"

    return mark, adjusted_stop


def bin_regions_sharded(
        in_file,
        out_prefix,
        bin_size,
        stride,
        final_length,
        chromsizes,
        method="naive", # <- TODO deprecate? handle flanks elsewhere?
        num_flanks=3, # <- TODO deprecate? handle flanks elsewhere?
        max_size=1000000): # max size: total num of bins allowed in one file
    """Bin regions based on bin size and stride

    Args:
      in_file: BED file to bin
      out_file: name of output binned file
      bin_size: length of the bin 
      stride: how many base pairs to jump for next window
      method: how to do binning (add flanks or not)

    Returns:
      None
    """
    logging.info("binning regions for {}...".format(in_file))
    assert os.path.splitext(in_file)[1] == '.gz'
    extend_len = (final_length - bin_size) / 2
    
    total_bins_in_file = 0
    file_idx = 0
    out_file = "{}.{}.bed.gz".format(out_prefix, str(file_idx).zfill(3))
    # Open input and output files and bin regions
    with gzip.open(in_file, 'rb') as fp:
        for line in fp:
            fields = line.strip().split('\t')
            chrom, start, stop = fields[0], int(fields[1]), int(fields[2])
            if len(fields) > 3:
                metadata = "metadata={};".format(fields[3])
            else:
                metadata = ""

            # adjust start/stop as needed
            adjusted_start, adjusted_stop = _get_adjusted_start_and_stop(
                start, stop, method, num_flanks=num_flanks)

            # bin and save out
            while adjusted_start < adjusted_stop:
                region_id = "{}={}:{}-{}".format(
                    MetaKeys.REGION_ID, chrom, start, stop)
                active_id = "{}={}:{}-{}".format(
                    MetaKeys.ACTIVE_ID, chrom, adjusted_start, adjusted_start + bin_size)
                features_id = "{}={}:{}-{}".format(
                    MetaKeys.FEATURES_ID,
                    chrom,
                    adjusted_start - extend_len,
                    adjusted_start - extend_len + final_length)
                
                # write out bins
                # TODO one way to speed this up is to keep in mem until file change, only then write out
                with gzip.open(out_file, 'a') as out:
                    out.write((
                        "{0}\t{1}\t{2}\t"
                        "{3}{4};{5};{6}\n").format(
                            chrom,
                            adjusted_start,
                            adjusted_start + bin_size,
                            metadata,
                            region_id,
                            active_id,
                            features_id))

                # increment start position and total bins
                adjusted_start += stride
                total_bins_in_file += 1

                # if max size hit, reset and go to new file
                if total_bins_in_file >= max_size:
                    file_idx += 1
                    out_file = "{}.{}.bed.gz".format(out_prefix, str(file_idx).zfill(3))
                    total_bins_in_file = 0

    # TODO deal with this elsewhere? works if flanks are deprecated here
    # and then check chrom boundaries - throw away bins that fall outside the boundaries
    chrom_bed_files = glob.glob("{}[.]*.bed.gz".format(out_prefix))
    for bed_file in chrom_bed_files:
        filt_bed_file = "{}.filt.bed.gz".format(bed_file.split(".bed")[0])
        overlap_check = (
            "bedtools slop -i {0} -g {1} -b {2} | "
            "awk -F '\t' '{{ print $1\"\t\"$2+{2}\"\t\"$3-{2}\"\t\"$4 }}' | "
            "awk -F '\t' '{{ if ($3-$2=={3}) {{ print }} }}' | "
            "gzip -c > {4}").format(
                bed_file,
                chromsizes,
                extend_len,
                bin_size,
                filt_bed_file)
        logging.debug(overlap_check)
        os.system(overlap_check)

        # if the file is now empty, throw away
        with gzip.open(filt_bed_file, "r") as fp:
            data = fp.read(1)
        if len(data) == 0:
            os.system("rm {}".format(filt_bed_file))
        
    return None



def bin_regions_parallel(
        bed_files,
        out_dir,
        chromsizes,
        bin_size=200,
        stride=50,
        final_length=1000,
        parallel=12):
    """bin in parallel
    """
    split_queue = setup_multiprocessing_queue()
    
    for bed_file in bed_files:
        prefix = os.path.basename(bed_file).split(".narrowPeak")[0].split(".bed")[0]
        split_args = [
            bed_file,
            "{}/{}".format(out_dir, prefix),
            bin_size,
            stride,
            final_length,
            chromsizes,
            "naive"]
        split_queue.put([bin_regions_sharded, split_args])

    # run the queue
    run_in_parallel(split_queue, parallel=parallel, wait=True)

    return None


# TODO deprecate
def extract_active_centers_OLD(binned_file, fasta_file):
    """given a fasta file produced in this pipeline, extract out
    the active center 
    """
    bin_count = 0
    with gzip.open(binned_file, 'w') as out:
        with gzip.open(fasta_file, 'r') as fp:
            for line in fp:
                fields = line.strip().split()

                # extract bin only
                metadata = fields[0].split("::")[0]
                metadata_dict = dict([field.split("=") for field in metadata.split(";")])
                region_bin = metadata_dict["region"]

                chrom = region_bin.split(":")[0]
                start = int(region_bin.split(':')[1].split('-')[0])
                stop = int(region_bin.split(':')[1].split('-')[1].split('.')[0])
                
                out.write('{}\t{}\t{}\t{}\n'.format(chrom, start, stop, metadata))
                bin_count += 1
    
    return bin_count


def select_flank_negatives(
        positives_bed_file,
        negatives_bed_file,
        bin_size,
        stride,
        num_flank_regions=3):
    """given a positives set, get negatives from the flanks
    """
    with gzip.open(positives_bed_file, "r") as fp:
        with gzip.open(negatives_bed_file, "w") as out:
            for line in fp:
                fields = line.strip().split('\t')
                chrom, start, stop = fields[0], int(fields[1]), int(fields[2])

                # get left flanks
                left_start = max(start - num_flank_regions * stride, 0)
                left_stop = start

                # get right flanks
                right_start = stop
                right_stop = stop + num_flank_regions * stride

                # write out
                metadata = "region={0}:{1}-{2};negative_type=flank_neg".format(
                    chrom, start, stop)
                out.write("{0}\t{1}\t{2}\t{3}\n".format(
                    chrom, left_start, left_stop, metadata))
                out.write("{0}\t{1}\t{2}\t{3}\n".format(
                    chrom, right_start, right_stop, metadata))

    # TODO slopbed?
                
    return None


def select_negatives_from_region_set(
        positives_bed_file,
        superset_bed_file,
        neg_region_num,
        negatives_bed_file):
    """given a superset bed file, select a number of regions
    """
    select_negs = (
        "bedtools intersect -v -a {0} -b {1} | "
        "shuf -n {2} --random-source={1} | "
        "awk '{{ print $1\"\t\"$2\"\t\"$3 }}' | "
        "gzip -c > {3}").format(
            positives_bed_file,
            superset_bed_file,
            neg_region_num,
            negatives_bed_file)
    print select_negs
    os.system(select_negs)

    return None


def select_random_negatives(
        positives_bed_file,
        negatives_bed_file,
        chrom_sizes,
        neg_region_num,
        tmp_dir="."):
    """select random negatives from across teh genome
    """
    # set up chrom sizes - remove chrM
    tmp_chrom_sizes = "{0}/{1}.tmp".format(tmp_dir, os.path.basename(chrom_sizes))
    setup_chrom_sizes = (
        "cat {0} | grep -v '_' | grep -v 'chrM' > "
        "{1}").format(chrom_sizes, tmp_chrom_sizes)
    print setup_chrom_sizes
    os.system(setup_chrom_sizes)

    # get total region num from positives bed file
    num_positive_regions = 0
    with gzip.open(positives_bed_file, "r") as fp:
        for line in fp:
            num_positive_regions += 1
            
    # select random negatives
    # this uses bedtools shuffle, which maintains distribution of lengths
    # of the regions. as such, it relies on the positives bed file
    # if you are selecting more negatives than there are total regions,
    # you'll run this multiple times to maintain the lengths distribution
    # as much as possible.
    random_neg_left = neg_region_num
    while random_neg_left > 0:
        random_negs_to_select = min(random_neg_left, num_positive_regions)
        select_negs = (
            "bedtools shuffle -i {0} -excl {0} -g {1} -seed 42 | "
            "head -n {2} | "
            "gzip -c >> {3}").format(
                positives_bed_file,
                tmp_chrom_sizes,
                random_negs_to_select,
                negatives_bed_file)
        print select_negs
        os.system(select_negs)
        random_neg_left -= random_negs_to_select
    
    return None


def select_all_negatives(
        positives_bed_file,
        negatives_bed_file,
        chrom_sizes,
        tmp_dir="."):
    """select all negatives outside of the positives
    """
    # set up chrom sizes - remove chrM
    tmp_chrom_sizes = "{0}/{1}.tmp".format(tmp_dir, os.path.basename(chrom_sizes))
    setup_chrom_sizes = (
        "cat {0} | grep -v '_' | sort -k1,1 -k2,2n | grep -v 'chrM' > "
        "{1}").format(chrom_sizes, tmp_chrom_sizes)
    print setup_chrom_sizes
    os.system(setup_chrom_sizes)

    # get complement
    select_negs = (
        "bedtools complement -i {0} -g {1} | "
        "gzip -c > {2}").format(
            positives_bed_file,
            tmp_chrom_sizes,
            negatives_bed_file)
    print select_negs
    os.system(select_negs)
    
    return None


def setup_negatives(
        positives_bed_file,
        dhs_bed_file,
        chrom_sizes,
        bin_size=200,
        stride=50,
        genome_wide=True,
        tmp_dir="."):
    """wrapper to set up reasonable negative sets 
    for various needs
    """
    # set up
    prefix = "{}/{}".format(
        tmp_dir,
        os.path.basename(positives_bed_file).split(".bed")[0])
    num_positive_regions = 0
    with gzip.open(positives_bed_file, "r") as fp:
        for line in fp:
            num_positive_regions += 1
    
    # flank negatives
    num_flank_regions = 3
    flank_negatives_bed_file = "{}.flank-negatives.bed.gz".format(
        prefix, num_flank_regions)
    select_flank_negatives(
        positives_bed_file,
        flank_negatives_bed_file,
        bin_size,
        stride,
        num_flank_regions=num_flank_regions)

    # DHS negatives
    dhs_negatives_bed_file = "{}.dhs-negatives.bed.gz".format(prefix)
    select_negatives_from_region_set(
        positives_bed_file,
        dhs_bed_file,
        int(num_positive_regions/2.),
        dhs_negatives_bed_file)

    # also infuse random negatives
    random_negatives_bed_file = "{}.random-negatives.bed.gz".format(prefix)
    select_random_negatives(
        positives_bed_file,
        random_negatives_bed_file,
        chrom_sizes,
        int(num_positive_regions/2.))

    # now merge all these to get training negative set
    training_negatives_bed_file = "{}.training-negatives.bed.gz".format(prefix)
    merge_cmd = (
        "zcat {0} {1} {2} | "
        "awk -F '\t' '{{print $1\"\t\"$2\"\t\"$3}}' | "
        "sort -k1,1 -k2,2n | "
        "bedtools merge -i stdin | "
        "gzip -c > {3}").format(
            flank_negatives_bed_file,
            dhs_negatives_bed_file,
            random_negatives_bed_file,
            training_negatives_bed_file)
    print merge_cmd
    os.system(merge_cmd)

    genomewide_negatives_bed_file = "{}.genomewide-negatives.bed.gz".format(prefix)
    if genome_wide:
        # then also set up genomic negatives (for evaluation)
        select_all_negatives(
            positives_bed_file,
            genomewide_negatives_bed_file,
            chrom_sizes)
        
    return training_negatives_bed_file, genomewide_negatives_bed_file


def _build_intersect_cmd(
        label_bed_file, bed_file, out_tmp_file, method, chromsizes, extend_len=1000):
    """build the intersect command (uses bedtools) to get labels
    """
    if method == 'summit': # Must overlap with summit
        intersect = (
            "zcat {0} | "
            "awk -F '\t' '{{print $1\"\t\"$2+$10\"\t\"$2+$10+1}}' | "
            "bedtools intersect -c -a {1} -b stdin | "
            "gzip -c > "
            "{2}").format(label_bed_file, bed_file, out_tmp_file)
    elif method == 'half_peak': # Bin must be 50% positive
        intersect = (
            "bedtools intersect -f 0.5 -c "
            "-a <(zcat {0}) "
            "-b <(zcat {1}) | "
            "gzip -c > "
            "{2}").format(bed_file, label_bed_file, out_tmp_file)
    elif method == "histone_overlap":
        # label file is histone, so look for nearby histone mark
        intersect = (
            "bedtools slop -i {0} -g {1} -b {2} | "
            "bedtools intersect -c "
            "-a <(zcat {3}) "
            "-b stdin | "
            "gzip -c > "
            "{4}").format(
                label_bed_file,
                chromsizes,
                extend_len,
                bed_file,
                out_tmp_file)
    elif method == "counts":
        # TODO(dk) do a counts version (for regression)
        # requires BED files of reads (tagAlign)
        raise ValueError, "method not yet implemented!"
    else:
        raise ValueError,  "method not yet implemented!"

    return intersect


def _setup_file_metadata(label_bed_files):
    """pull filenames and condense for readability/space constraints
    """
    # get metadata
    file_metadata = [
        "index={0};file={1}".format(
            i, os.path.basename(label_bed_files[i]).split(".bed")[0].split("narrowPeak")[0])
        for i in xrange(len(label_bed_files))]

    # check bytes
    if np.array(file_metadata).nbytes >= 64000:
        file_metadata = ["metadata too large"]

    return file_metadata


def generate_labels(
        bed_file,
        label_bed_files,
        key,
        h5_file,
        method="half_peak",
        chromsizes=None,
        tmp_dir="."):
    """given a bed file and label files, intersect 
    and return an array of labels
    """
    os.system("mkdir -p {}".format(tmp_dir))
    num_files = len(label_bed_files)
    file_metadata = _setup_file_metadata(label_bed_files)

    # for each label bed file
    for i in xrange(len(label_bed_files)):
        label_bed_file = label_bed_files[i]
        assert label_bed_file.endswith(".gz"), "{} is not gzipped".format(label_bed_file)
        out_tmp_file = "{}/{}.intersect.bed.gz".format(
            tmp_dir,
            os.path.basename(label_bed_file).split(".narrowPeak")[0].split(".bed.gz")[0])
        
        # Do the intersection to get a series of 1s and 0s
        intersect = _build_intersect_cmd(
            label_bed_file, bed_file, out_tmp_file, method, chromsizes)
        logging.debug('{0}: {1}'.format(bed_file, intersect))
        os.system('GREPDB="{0}"; /bin/bash -c "$GREPDB"'.format(intersect))
        
        # read in the results
        bed_labels = pd.read_table(
            out_tmp_file,
            header=None,
            names=['Chr', 'Start', 'Stop', "pos"])

        # store in tmp array
        if i == 0:
            num_rows = bed_labels.shape[0]
            labels = np.zeros((num_rows, num_files), dtype=np.bool)
            labels[:,i] = (bed_labels["pos"] >= 1.0).astype(np.bool)
        else:
            labels[:,i] = (bed_labels["pos"] >= 1.0).astype(np.bool)

        # delete tmp file
        os.system("rm {}".format(out_tmp_file))

    # add data into h5 file
    with h5py.File(h5_file, "a") as hf:
        hf.create_dataset(key, dtype="u1", shape=labels.shape)
        logging.debug("storing label set in file")
        for i in range(num_rows):
            if i % 10000 == 0:
                logging.debug("{}".format(i))
            hf[key][i,:] = labels[i,:]
        hf[key].attrs[AttrKeys.FILE_NAMES] = file_metadata
                
    return None
