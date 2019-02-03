import torch
from torch.autograd import Variable
from utils.data_utils import *
from utils.fetch_global_args import ALL_CHROMOSOMES
import importlib.util
import numpy as np
import pdb


def integrated_gradients(model, input_list, steps=50, baseline=None, output_extract_fn=lambda x: x, max_batch_size=128,
                         **kwargs):
    """ Uses the integrated gradients method from
        https://arxiv.org/pdf/1703.01365.pdf to compute attribution scores.

    Args:
        model: any function that operates on the input_list in a batch-wise
            fashion (may be nn.Module, in which case call model.eval() before
            passing in)
        input_list: a list of Tensors that are inputs to the model, first dim
            (batch_size) must be 1, i.e. input one example at a time.
        steps: number of steps to approximate integral. Default 50.
        baseline: baseline input as defined in the paper. Default is zeros. If
            not None, must be a list of Tensors matching input_list
        output_extract_fn: Defines how to extract the output of interest for a
            batch of inputs. Default is identity fn.

            E.g if model has a single output with a prob over 10 classes, then
            output dim is batch_size x 10. If output of interest is the 7th
            class, then output_extract_fn= lambda x: x[:,7]. If the model has
            2 distinct outputs in which the 0th output is a prob over 10 classes,
            then the analogous output_extract_fn= lambda x: x[0][:,1]

        max_batch_size: Max batch size for one model forward and backward.
            Default 128.

    """
    cuda = input_list[0].is_cuda
    num_inps = len(input_list)

    if not baseline:
        # use a zero baseline
        baseline = [torch.zeros(inp.size()) for inp in input_list]
        if cuda: baseline = [inp.cuda() for inp in baseline]

    for i in range(num_inps):
        assert (input_list[i].size(0) == 1)  # num_batches=1 and dim 0 is batch dim

        input_list[i] -= baseline[i]

        baseline[i] = torch.cat([baseline[i]] * steps)  # copying it step times along first dim
        for j in range(steps):
            baseline[i][j] += ((j + 1) / float(steps)) * input_list[i][0]

    ig = [torch.squeeze(torch.zeros(inp.size()), 0) for inp in input_list]  # removing first batch dim
    if cuda: ig = [inp.cuda() for inp in ig]

    for b in range(0, steps, max_batch_size):
        batch = [Variable(inp[b:b + max_batch_size], requires_grad=True) for inp in baseline]

        # extract output from model (single scalar)
        out = torch.sum(output_extract_fn(model(*batch)))

        # compute gradients
        out.backward()

        for i in range(num_inps):
            ig[i] += torch.sum(batch[i].grad.data, dim=0)

    return [(torch.squeeze(input_list[i], 0) * ig[i] / steps).cpu().numpy() for i in range(num_inps)]


def mutate_input(model, input_list, input_of_interest=0, output_extract_fn=lambda x: x, mutate_val=0,
                 return_diff=True, max_batch_size=128, **kwargs):
    """ Switches each value in input_list[input_of_interest].size(1) to mutate_val
        one-by-one and collects output_extract_fn(model(<input>)).

    Args:
        model: any function that operates on the input_list in a batch-wise
            fashion (may be nn.Module, in which case call model.eval() before
            passing in)
        input_list: a list of Tensors that are inputs to the model, first dim
            (batch_size) must be 1, i.e. input one example at a time.

        input_of_interest: Chooses index of input to mutate. Only one at a time.
            Default is 0.

            E.g. if a model takes 2 inputs (len(input_list)==2), and input of
            interest to mutate is the 1st, then input_of_interest=1

        output_extract_fn: Defines how to extract the output of interest for a
            batch of inputs. Should extract a vector (by choosing one scalar for
            each input). Default is identity fn.

            E.g if model has a single output with a prob over 10 classes, then
            output dim is batch_size x 10. If output of interest is the 7th
            class, then output_extract_fn= lambda x: x[:,7]. If the model has
            2 distinct outputs in which the 0th output is a prob over 10 classes,
            then the analogous output_extract_fn= lambda x: x[0][:,1]

        mutate_val: The value to which each position in the 1st dim of the input
            of interest is mutated to.

        return_diff: If True, return the difference from non-mutated output. Default
            is True.
 
        max_batch_size: Max batch size for one model forward and backward.
            Default 128.

    Returns:
        A numpy vector of length input_list[input_of_interest].size(1) with values of
        output_extract_fn(model(<input>)) for mutated inputs.

    """

    cuda = input_list[0].is_cuda
    num_inps = len(input_list)
    num_feats = input_list[input_of_interest].size(1)

    for i in range(num_inps):
        assert (input_list[i].size(0) == 1)  # num_batches=1 and dim 0 is batch dim

        input_list[i] = torch.cat([input_list[i]] * (num_feats + 1))  # last one for original input

        if i == input_of_interest:
            for j in range(num_feats):
                input_list[i][j][j] = mutate_val

    outputs = []

    for b in range(0, num_feats + 1, max_batch_size):
        batch = [Variable(inp[b:b + max_batch_size], requires_grad=False) for inp in input_list]

        # extract output from model (a vector)
        out = output_extract_fn(model(*batch))

        # ASSUMES out is a vector
        out = out.view(-1)
        outputs.append(out.cpu().data.numpy())

    outputs = np.concatenate(outputs)

    if return_diff: outputs[:-1] -= outputs[-1]

    return outputs[:-1]


def grad_x_input(model, input_list, output_extract_fn=lambda x: x, mask_with_input=True, max_batch_size=128, **kwargs):
    """ Uses gradient x input to compute attribution scores.

    Args:
        model: any function that operates on the input_list in a batch-wise
            fashion (may be nn.Module, in which case call model.eval() before
            passing in)
        input_list: a list of Tensors that are inputs to the model, first dim
            is num_inputs and can be >=1.
        output_extract_fn: Defines how to extract the output of interest for a
            batch of inputs. Default is identity fn.

            E.g if model has a single output with a prob over 10 classes, then
            output dim is batch_size x 10. If output of interest is the 7th
            class, then output_extract_fn= lambda x: x[:,7]. If the model has
            2 distinct outputs in which the 0th output is a prob over 10 classes,
            then the analogous output_extract_fn= lambda x: x[0][:,1]


        mask_with_input: Multiply gradients by input (True by default, added for cases
            when pure gradients are required).


        max_batch_size: Max batch size for one model forward and backward.
            Default 128.

    Returns:
        grad_x_input: a list of same elements as input_list, with grad x input
            for each input.

    """

    cuda = input_list[0].is_cuda
    num_inps = len(input_list)
    gxi = [[] for _ in range(len(input_list))]

    for b in range(0, input_list[0].size(0), max_batch_size):
        batch = [Variable(inp[b:b + max_batch_size], requires_grad=True) for inp in input_list]

        # extract output from model (single scalar)
        out = torch.sum(output_extract_fn(model(*batch)))

        # compute gradients
        out.backward()

        for i in range(num_inps):
            gxi[i].append(batch[i].grad.data)

    if mask_with_input:
        return [(torch.cat(gxi[i])*input_list[i]).cpu().numpy() for i in range(num_inps)]
    else:
        return [torch.cat(gxi[i]).cpu().numpy() for i in range(num_inps)]
