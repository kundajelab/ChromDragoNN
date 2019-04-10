# description: preprocess FASTA sequences

import os
import re
import sys
import gzip
import glob
import h5py
import logging
import subprocess

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from parallelize import setup_multiprocessing_queue
from parallelize import run_in_parallel

from subprocess import Popen, PIPE, STDOUT


def one_hot_encode(sequence):
    """One hot encode sequence from string to numpy array

    Args:
      sequence: string of sequence, using [ACGTN] convention

    Returns:
      one hot encoded numpy array of sequence
    """
    # set for python version
    integer_type = np.int8 if sys.version_info[0] == 2 else np.int32

    # set up sequence array
    sequence_length = len(sequence)
    sequence_npy = np.fromstring(sequence, dtype=integer_type)

    # one hot encode
    integer_array = LabelEncoder().fit(
        np.array(('ACGTN',)).view(integer_type)).transform(
            sequence_npy.view(integer_type)).reshape(1, sequence_length)
    one_hot_encoding = OneHotEncoder(
        sparse=False, n_values=5).fit_transform(integer_array)

    return one_hot_encoding.reshape(
        1, 1, sequence_length, 5)[:, :, :, [0, 1, 2, 4]]


class GenomicIntervalConverter(object):
    """converts genomic intervals to sequence"""

    def __init__(self, lock, fasta, batch_size, seq_len=1000):
        """initialize the pipe
        requires a lock to make sure it's set up thread-safe
        """
        # initialize
        lock.acquire()
        pipe_in, pipe_out, close_fn = GenomicIntervalConverter.setup_converter(fasta)
        lock.release()

        # save
        self.converter_in = pipe_in
        self.converter_out = pipe_out
        self.close = close_fn

        # set up tmp numpy array so that it's not re-created for every batch
        self.onehot_batch_array = np.zeros((batch_size, seq_len), dtype=np.uint8)

        
    @staticmethod
    def setup_converter(fasta):
        """sets up the pipe to convert a sequence string to onehot
        NOTE: you must unbuffer things to make sure they flow through the pipe
        """
        # set up input pipe. feed using: pipe_in.write(interval), pipe_in.flush()
        pipe_in = subprocess.Popen(
            ["cat", "-u", "-"],
            stdout=PIPE, stdin=PIPE, stderr=STDOUT)

        # get fasta sequence
        get_fasta_cmd = [
            "bedtools",
            "getfasta",
            "-tab",
            "-fi",
            "{}".format(fasta),
            "-bed",
            "stdin"]
        get_fasta = subprocess.Popen(
            get_fasta_cmd,
            stdin=pipe_in.stdout, stdout=PIPE)

        # replace ACGTN with 01234 and separate with commas
        sed_cmd = [
            'sed',
            "-u",
            's/^.*[[:blank:]]//g; s/[Aa]/0/g; s/[Cc]/1/g; s/[Gg]/2/g; s/[Tt]/3/g; s/[Nn]/4/g; s/./,&/g; s/,//']
        pipe_out = subprocess.Popen(
            sed_cmd,
            stdin=get_fasta.stdout, stdout=PIPE)

        # set up close fn
        def close_fn():
            pipe_in.terminate()
            pipe_in.wait()
            get_fasta.wait()
            pipe_out.wait()

        return pipe_in, pipe_out, close_fn


    def convert(self, array, seq_len=1000):
        """given a set of intervals, get back sequence info
        """
        for i in xrange(array.shape[0]):

            metadata_dict = dict([
                val.split("=")
                for val in array[i][0].strip().split(";")])
            try:
                feature_interval = metadata_dict["features"].replace(
                    ":", "\t").replace("-", "\t")
                feature_interval += "\n"
                # pipe in and get out onehot
                self.converter_in.stdin.write(feature_interval)
                self.converter_in.stdin.flush()

                # check pipe 
                sequence = self.converter_out.stdout.readline().strip()

                # convert to array - this is the crucial speed step
                # must separate with commas, otherwise will not be read in correctly
                sequence = np.fromstring(sequence, dtype=np.uint8, sep=",")
            
            except:
                # exceptions shouldn't exist, if preprocessed correctly
                print "sequence information missing, {}".format(feature_interval)
                sequence = np.array([4 for j in xrange(seq_len)], dtype=np.uint8)

            # save out
            self.onehot_batch_array[i,:] = sequence

        return self.onehot_batch_array
