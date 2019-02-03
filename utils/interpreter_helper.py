import torch
from utils.data_utils import *
from utils.fetch_global_args import ALL_CHROMOSOMES
import numpy as np
import pdb

def place_motif(seq, motif):
    """ Places motif at the center of seq. Ignores empty flanking columns of motif.
    If motif is a PWM then takes the most likely sequence.
    
    seq: Tensor of size L x 4 
    motif: Tensor of size W x 4
    
    Returns seq with motif at the middle.
    """
    
    # trim motif 
    trimmed = motif[np.where(np.sum(motif,-1)!=0)[0]]
    trimmed = torch.Tensor((np.argsort(np.argsort(trimmed,-1),-1) == 3).astype(int))  # set largest to 1
    
    start_ind = (seq.size(0)-trimmed.size(0))//2
    seq[start_ind:start_ind+trimmed.size(0)] = trimmed
    return seq
    

def insert_motifs(model, input_list, motifs, output_extract_fn=lambda x: x[:, 1],
                  return_diff=True, max_batch_size=512, **kwargs):
    """ Places each motif in motifs at the center of each sequence and collects 
        output_extract_fn(model(<input>)).

    Args:
        model: any function that operates on the input_list in a batch-wise
            fashion (may be nn.Module, in which case call model.eval() before
            passing in)
        input_list: a list of Tensors that are inputs to the model, first dim
            (batch_size) must be 1, i.e. input one example at a time.

        motifs: Numpy array of shape N x W x 4, where N is the number of motifs, 
            W is the width. Individual motifs can have width <= W, and all 4 
            values should be set to 0 for flanking rows.

        output_extract_fn: Defines how to extract the output of interest for a
            batch of inputs. Should extract a vector (by choosing one scalar for
            each input). Default is identity fn.

            E.g if model has a single output with a prob over 10 classes, then
            output dim is batch_size x 10. If output of interest is the 7th
            class, then output_extract_fn= lambda x: x[:,7]. If the model has
            2 distinct outputs in which the 0th output is a prob over 10 classes,
            then the analogous output_extract_fn= lambda x: x[0][:,1]

        return_diff: If True, return the difference from raw sequence output. Default
            is True.
                  
        max_batch_size: Max batch size for one model forward and backward.
            Default 512.

    Returns: (N,) numpy array with scores after placing each of the filters in
            the center of the sequence.

    """

    cuda = input_list[0].is_cuda
    num_inps = len(input_list)
    num_motifs = motifs.shape[0]
    SEQ_INP_INDEX = 0   #  sequence is the input_list[0] 

    for i in range(num_inps):
        assert (input_list[i].size(0) == 1)  # num_batches=1 and dim 0 is batch dim

        input_list[i] = torch.cat([input_list[i]] * (num_motifs + 1))  # last one for original input

        if i == SEQ_INP_INDEX:
            for j in range(num_motifs):
                input_list[i][j] = place_motif(input_list[i][j], motifs[j])

    outputs = []

    for b in range(0, num_motifs + 1, max_batch_size):
        batch = [inp[b:b + max_batch_size] for inp in input_list]

        # extract output from model (a vector)
        out = output_extract_fn(model(*batch))

        # ASSUMES out is a vector
        out = out.view(-1)
        outputs.append(out.cpu().data.numpy())

    outputs = np.concatenate(outputs)

    if return_diff: outputs[:-1] -= outputs[-1]

    return outputs[:-1]
