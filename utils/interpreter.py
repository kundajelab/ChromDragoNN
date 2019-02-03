# loads all chromosomes to memory (~ 20Gb)
import joblib
from tqdm import tqdm
import os
import numpy as np
import time
import math
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.attribute import integrated_gradients, mutate_input, grad_x_input
from utils.interpreter_helper import insert_motifs
import pdb
from utils.data_utils import *

flatten = lambda x: [l for c in x for l in c]


class Interpreter:

    def __init__(self, model, args, di, regions, window=0, subsampled=500):
        """

        :param model: The model to evaluate on
        :param args: arguments used in instantiation of model
        :param di: data iterator used for the model
        :param regions: A dictionary of dictionary of lists denoting the regions of the genome to interpret on
        The outer keys of the dictioanry are the cell-types to impute on, the inner keys are the chromsones,
        and the inner values are indices (hereafter referred to as anchors) into the subsampled data
        for which you want a predictions

        :example regions:
                regions = {0: {1 : [1,2,10]}, 1: {1: [4,5,10], 2: [5,6,10]}}
                This would specify that you want to impute predictions on
                Cell-Type 0, Chromosone 1, indices 1,2,10
                Cell-Type 1, Chromosone 1, indices 4,5,10
                Cell-Type 1, Chromosone 2, indices 5,6,10

        :param window: Integer, the window around the anchor for which to impute on
        :param subsampled: The original subsampling of the data, i.e. the number of bins between
            anchor 1 and anchor 2 in the sequence data within the dataiterator

        """
        self.model = model
        self.args = args
        self.di = di
        self.avail_regions = {}

        self.regions = regions
        self.window = window
        self.num_examples = 0
        self.min_batches = 0
        self.num_anchors = 0

        for ctype, chrms in self.regions.items():
            self.avail_regions[ctype] = {}

            for chrm, idxs in chrms.items():
                c_start, c_end = self.di.dnase_chrm_range[chrm]

                indices = [list(range(c_start + x * subsampled - window, c_start + x * subsampled + window + 1)) for x
                           in idxs]
                indices = flatten(indices)

                self.avail_regions[ctype][chrm] = indices
                self.num_examples += len(indices)
                self.min_batches += int(math.ceil(len(indices) / self.args.batch_size))
                self.num_anchors += len(idxs)

        if self.args.cuda:
            self.model.cuda()
        else:
            self.model.cpu()

    def _eval_generator(self, batch_size, eval_subsample=1):
        """
        Generator for interpret data. Goes cell_type by cell_type in the regions,  chrm by chrms within a cell-tye,
        Subsamples by factor eval_subsample
        Unpacks labels and one-hot encoded sequences. Returns the data without labels.

        Input : batch_size, eval_subsample,

        Note: if eval_subsample == self.window, then the function returns batches only at the anchor points

        Yields: (ctype, chrm, indices), (sequence, gene_exp, locus_mean) or (ctype, chrm, index), (sequence, gene_exp)
            - ctype        string indicating cell-type of current batch
            - chrm         string indicating chrm of current batch
            - indices      list of indices of the current batch
            - sequence:    numpy array of dim (batch_size, 1000, 4), one hot encoded with 4 channels (ATCG)
            - gene_exp:    numpy array of dim (batch_size, num_genes)
            - locus_mean:  numpy array of dim (batch_sie)   [if self.return_locus_mean]


        """
        print("eval_subsample is {}".format(eval_subsample))

        for ctype, chrms in self.regions.items():
            for chrm in chrms:
                indices = self.avail_regions[ctype][chrm]
                if eval_subsample == self.window:
                    reduced_indices = indices[eval_subsample::2 * eval_subsample + 1]
                else:
                    reduced_indices = indices[::eval_subsample]
                if batch_size == -1:
                    batch_data = self.di._fetch_seq_genexp_batch(np.array(reduced_indices), np.array([ctype] * len(reduced_indices))) \
                                 + self.di.return_locus_mean * (self.di.dnase_label_mean[reduced_indices],)
                    yield (ctype, chrm, reduced_indices  ), batch_data
                else:
                    for i in range(0, len(reduced_indices), batch_size):
                        seq_inds = reduced_indices[i:i + batch_size]
                        batch_data = self.di._fetch_seq_genexp_batch(np.array(seq_inds), np.array([ctype] * len(seq_inds))) \
                                     + self.di.return_locus_mean * (self.di.dnase_label_mean[seq_inds],)
                        yield (ctype, chrm, seq_inds), batch_data

    def __form_input__(self, batch_info, impute=False):

        seq_batch, gene_batch, *locus_mean_batch = batch_info
        # unpack the index info for easy saving

        # ensure this chrm exists in the ditionary

        seq_batch = torch.from_numpy(seq_batch)
        gene_batch = torch.FloatTensor(gene_batch)

        # If model is testing with mean

        if locus_mean_batch:
            locus_mean_batch = torch.FloatTensor(locus_mean_batch[0])

            if self.args.cuda:
                seq_batch, gene_batch, locus_mean_batch = seq_batch.contiguous().cuda(), gene_batch.contiguous().cuda(), locus_mean_batch.contiguous().cuda()

            if impute:
                seq_batch = Variable(seq_batch, volatile=True)
                gene_batch = Variable(gene_batch, volatile=True)
                locus_mean_batch = Variable(locus_mean_batch, volatile=True)

                inputs = seq_batch, gene_batch, locus_mean_batch

            else:
                inputs = [[seq_batch[i:i + 1], gene_batch[i:i + 1], locus_mean_batch[i:i + 1]] for i in
                          range(seq_batch.shape[0])]


        else:
            if self.args.cuda:
                seq_batch, gene_batch = seq_batch.contiguous().cuda(), gene_batch.contiguous().cuda()

            if impute:
                seq_batch, gene_batch = Variable(seq_batch, volatile=True), Variable(gene_batch, volatile=True)
                inputs = seq_batch, gene_batch

            else:
                inputs = [[seq_batch[i:i + 1], gene_batch[i:i + 1]] for i in range(seq_batch.shape[0])]

        return inputs

    def impute_regions(self, retain_labels=False, write_to_file=False):
        """
        Evaluates the model on regions of the genome specified in self.regions. Returns a three level dictionary,
        with the outer keys as cell-types, the 2nd level keys as chromosones, and the inner keys as 'preds'
        or ['preds', 'labels'] depending on if retain_labels is true:

        Example Usage:
            regions = {0: {1 : [0,2,10]}}

            intp = Interpreter(model, args, di, results)
            res = intp.impute_regions()

            res := {0: {1 : {'preds': [array of shape 2*window + 1,
                                       array of shape 2*window + 1,
                                       array of shape 2*window + 1]}}

        :param retain_labels: whether or not to return the true labels corresponding to those indices
        :return:
        """

        # switch to evaluate mode
        self.model.eval()

        max_batches = int(math.ceil(self.num_examples / self.args.batch_size))
        # print(max_batches)
        # print(self.num_examples)
        max_batches = max(max_batches, self.min_batches)
        # print(max_batches)
        # print(self.min_batches)

        # create dictionary to store results
        results = {x: {} for x in self.avail_regions.keys()}

        for index_info, batch_info in tqdm(self._eval_generator(self.args.batch_size), total=max_batches):
            # unpack the index info for easy saving
            ctype, chrm, indices = index_info

            # ensure this chrm exists in the ditionary
            if not results[ctype].get(chrm, None):
                results[ctype][chrm] = {'preds': [], 'labels': []}

            model_inputs = self.__form_input__(batch_info, impute=True)
            # compute output
            outputs = self.model(*model_inputs)
            index = Variable(torch.LongTensor([1]))
            if self.args.cuda:
                index = index.cuda()

            preds = torch.index_select(outputs, 1, index=index).view(-1).cpu().data.numpy()
            # If we want to retain labels
            if retain_labels:
                labels = self.di._fetch_labels_batch(np.array(indices), np.array([ctype] * len(indices)))
                results[ctype][chrm]['labels'].extend(labels)

            # Save the results
            results[ctype][chrm]['preds'].extend(preds)

            # measure elapsed time

        # reformat the outer results
        final_results = {}
        for ctype, chrms in results.items():
            final_results[ctype] = {}
            for chrm in chrms:
                if retain_labels:
                    final_results[ctype][chrm] = {'preds': [], 'labels': []}
                else:
                    final_results[ctype][chrm] = {'preds': []}

                slice = lambda x: [x[i:i + 2 * self.window + 1] for i in range(0, len(x), 2 * self.window + 1)]
                final_results[ctype][chrm]['preds'] = slice(np.exp(results[ctype][chrm]['preds']))

                if retain_labels:
                    final_results[ctype][chrm]['labels'] = slice(results[ctype][chrm]['labels'])

        # save
        if write_to_file:
            predictions_path = os.path.join(self.args.checkpoint, 'predictions')
            print('Regional Imputation Done, saving in {}'.format(predictions_path))

            if not os.path.exists(predictions_path):
                os.mkdir(predictions_path)

            joblib.dump(final_results, os.path.join(predictions_path, 'interpret_results.joblib'))

        return final_results

    def __interpretWrapper__(self, eval_subsample, batch_size, label):

        def real_decorator(function):

            def wrapper(self, *args, **kwargs):
                self.model.eval()

                if eval_subsample == self.window and batch_size == 1:
                    max_batches = self.num_anchors
                else:
                    max_batches = int(math.ceil(self.num_examples / eval_subsample) / batch_size)

                final_exponentiate = kwargs.get('final_exponentiate', False)

                if final_exponentiate:
                    output_extract_fn = lambda y: torch.exp(y[:, 1])
                else:
                    output_extract_fn = lambda y: y[:, 1]

                kwargs.update({'output_extract_fn': output_extract_fn})


                # create dictionary to store results
                results = {x: {} for x in self.avail_regions.keys()}
                for index_info, batch_info in tqdm(self._eval_generator(batch_size, eval_subsample), total=max_batches):
                    # unpack the batch info
                    ctype, chrm, indices = index_info

                    if not results[ctype].get(chrm, None):
                        results[ctype][chrm] = {label: []}

                    inputs = self.__form_input__(batch_info)
                    kwargs.update({'inputs': inputs})
                    res = function(**kwargs)
                    results[ctype][chrm][label].extend(res)

                return results

            return wrapper

        return real_decorator

    def  _get_arg(self, argname, **kwargs):
        assert (argname in kwargs.keys())         
        return kwargs.get(argname)
        
    def IG_step(self, **kwargs):
        attribution_data = [integrated_gradients(self.model, x, **kwargs) for x in self._get_arg('inputs', **kwargs)]
        return attribution_data

    def mutation_step(self, **kwargs):
        mutated_data = [mutate_input(self.model, x, **kwargs) for x in self._get_arg('inputs', **kwargs)]
        return mutated_data

    def motif_insertion_step(self, **kwargs):
        inserted_data = [insert_motifs(self.model, x, **kwargs) for x in self._get_arg('inputs', **kwargs)]
        return inserted_data
        
    def GxI_step(self, **kwargs):
        inputs = self._get_arg('inputs', **kwargs)
        inputs = [torch.cat([x[z] for x in inputs], 0) for z in range(len(inputs[0]))]
        #print(inputs[0].size())
        res = grad_x_input(self.model, inputs, **kwargs)
        return res
        

    def perform_GxI(self, eval_subsample=1, batch_size=-1, label='GxI', **kwargs):
        """
        Runs integrated gradients on regions of interest within the genome, returns a three level dictionary,
        outer dictionary keys are cell-types, second-level keys --> chromosomes, and lowest level key 'IG'

        The values in the lowest level dictionary correspond the result of integrated gradients for that
        region in the genome. In our case this is a list of size two with results for both the sequence
        and the gene expression data

        Example Usage:
            regions = {0: {1 : [0,2,10]}}

            intp = Interpreter(model, args, di, regions)
            res = intp.perform_IG(1, self.window)

            res := {0: {1 : {'IG': [[IG_SEQ1, IG_RNA1], [IG_SEQ2, IG_RNA2], [IG_SEQ4, IG_RNA4]]}}


        :param batch_size: The batch_size (set to -1 unless you know what you are doing)
        :param eval_subsample: The subsample with which to evaluate. If only the anchor points are wanted set
                eval_subsample = self.window
        :kwargs
        :param steps:  Number of steps in the integrated gradients scheme
        :param final_exponentiate: If output should be exponentiated using torch.exp (use if output is in log space)
        :return:
        """
        return self.__interpretWrapper__(eval_subsample, batch_size, label)(self.GxI_step)(self, **kwargs)

    def perform_IG(self, eval_subsample=1, batch_size=1, label='IG', **kwargs):
        """
        Runs integrated gradients on regions of interest within the genome, returns a three level dictionary,
        outer dictionary keys are cell-types, second-level keys --> chromosomes, and lowest level key 'IG'

        The values in the lowest level dictionary correspond the result of integrated gradients for that
        region in the genome. In our case this is a list of size two with results for both the sequence
        and the gene expression data

        Example Usage:
            regions = {0: {1 : [0,2,10]}}

            intp = Interpreter(model, args, di, regions)
            res = intp.perform_IG(1, self.window)

            res := {0: {1 : {'IG': [[IG_SEQ1, IG_RNA1], [IG_SEQ2, IG_RNA2], [IG_SEQ4, IG_RNA4]]}}


        :param batch_size: The batch_size (set to 1 if only want anchor points)
        :param eval_subsample: The subsample with which to evaluate. If only the anchor points are wanted set
                eval_subsample = self.window
        :kwargs
        :param steps:  Number of steps in the integrated gradients scheme
        :param final_exponentiate: If output should be exponentiated using torch.exp (use if output is in log space)
        :return:
        """
        res =  self.__interpretWrapper__(eval_subsample, batch_size, label)(self.IG_step)(self, **kwargs)
        
        for ctype, chrms in self.regions.items():
            for chrm in chrms:
                res[ctype][chrm][label] = [np.vstack([np.expand_dims(x[i],0) for x in res[ctype][chrm][label]]) for i in range(len(res[ctype][chrm][label][0]))]
                
        return res

    def perform_genexp_mutation(self, eval_subsample=0, batch_size=1, label='genexp_diff', **kwargs):
        """
        Mutates gene expression one-by-one to on regions of interest within the genome and returns the
        difference from non-mutated input, returns a three level dictionary, outer dictionary keys are
        cell-types, second-level keys --> chromosomes, and lowest level key 'genexp_diff'

        The values in the lowest level dictionary correspond 'the result of integrated gradients for that
        region in the genome. In our case this is a list of size two with results for both the sequence
        and the gene expression data

        Example Usage:
            regions = {0: {1 : [0,2,10]}}

            intp = Interpreter(model, args, di, regions)
            res = intp.perform_genexp_mutation(self.window, final_exponentiate=True)    # if model output is log

            res := {0: {1 : {'genexp_diff': array([GENEXP_DIFF1, GENEXP_DIFF2, GENEXP_DIFF3])}}

        :param batch_size: The batch_size for generator (set to 1 if only want anchor points)
        :param eval_subsample: The subsample with which to evaluate. If only the anchor points are wanted set
                eval_subsample = self.window
        :param final_exponentiate: If output should be exponentiated using torch.exp (use if output is in log space)
        :return:
        """

        res = self.__interpretWrapper__(eval_subsample, batch_size, label)(self.mutation_step)(self, **kwargs)
        for ctype, chrms in self.regions.items():
            for chrm in chrms:
                # only one input which is input_of_interest
                res[ctype][chrm][label] = [np.vstack([np.expand_dims(x, 0) for x in  res[ctype][chrm][label]])]

        return res


    def perform_motif_insertion(self, label='motif_insertion', **kwargs):
        """
        Places each motif in motifs at the center of each sequence and computes the predicted accessibility. returns 
        a three level dictionary, outer dictionary keys are cell-types, second-level keys --> chromosomes, and lowest 
        level key label ('motif_insertion').
        
        Example Usage:
            regions = {0: {1: [0,2,10]}}
        
            intp = Interpreter(model, args, di, regions)
            res = intp.perform_motif_insertion(motifs=motifs, return_diff=True, final_exponentiate=True)  # if model output is log
                
            res := {0: {1: {'motif_insertion': arr}}}, where arr is 3 x N, N is the number of motifs and 3 corresponds
                                                       to number of locii (len([0,2,10]))
        
        :param motifs: Numpy array of shape N x W x 4, where N is the number of motifs, W is the width. Individual motifs
                       can have width <= W, and all 4 values should be set to 0 for flanking rows.
        """
        res = self.__interpretWrapper__(eval_subsample=0, batch_size=1, label=label)(self.motif_insertion_step)(self, **kwargs)
        
        for ctype, chrms in self.regions.items():
            for chrm in chrms:
                # making it NUM_LOCII x NUM_MOTIFS
                res[ctype][chrm][label] = [np.vstack([np.expand_dims(x, 0) for x in  res[ctype][chrm][label]])]

        return res
