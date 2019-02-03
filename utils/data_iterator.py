import joblib
import os
import numpy as np
import sklearn.metrics
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
from random import randint
import pdb



class DataIterator:
    def __init__(self, dnase_folder_path, rna_quants_file_path, hold_out, validation_list, test_list,
                 chromosomes = list(range(1,23)) + ['X', 'Y'], eval_subsample=500,
                 balance_classes_train=False, positive_proportion = .5, return_locus_mean=False):
        """
        Inputs: dnase_folder, rna_quants_file, hold_out, validation_list, test_list
            - dnase_folder_path:    path to folder that has dnase accessibility data for all chromosomes,
                                    packbited (e.g. dnase.chr21.packbit.joblib)
            - rna_quants_file_path: path to joblib file that has a (num_cell_types x num_genes)
                                    gene expression data (rna_quants.joblib)
            - hold_out:             'chromosomes' or 'cell_types'- choose what needs to be held out for
                                    evaluation
            - validation_list:      elements in validation_set; if 'chromosomes' are held out, this list will
                                    contain numbers from list(range(1,23)) + ['X','Y']; if 'cell_types' then
                                    numbers from list(range(num_cell_types))
            - test_list:            elements in test_set, specification same as for validation_list
            - eval_subsample:       factor for subsampling images for the methods eval_generator and evaluate_model
        """
    
        self.hold_out = hold_out
        self.validation_list = validation_list
        self.test_list = test_list
        self.eval_subsample = eval_subsample
        self.positive_prop = positive_proportion
        self.balance_classes_train = balance_classes_train
        self.return_locus_mean = return_locus_mean
        self.CHRMS = chromosomes
        
        self.load_dnase(dnase_folder_path)
        self.load_rna_quant(rna_quants_file_path)
        self.num_cell_types = self.rna_quants.shape[0]
        
        print('DATA ITERATOR ::: LOADED DATA')
        print('DATA ITERATOR ::: CELL TYPES : ' + str(self.rna_quants.shape[0]))

        self.prepare_blocks()
        
        
    def load_dnase(self, dnase_folder_path):
        """
        Loads all chromosomes and joins into one big array of size (num_examples, 1000, 1). Ignores metadata.

        Updates:
            - self.dnase_data:       of size (num_examples, 1000, 1) ; last dimension has been packbit-ed
            - self.dnase_labels:     of size (num_examples, num_cell_types/8) ; last dimension has been packbit-ed
            - self.dnase_chrm_range: dictionary of tuples, maintains index where each chromosome starts and ends
                                     (int for 1-22, else string)
        """
        
        self.dnase_data_raw = []
        self.dnase_labels_raw = []
        self.dnase_chrm_range = {}
        num_examples = 0
            
        for i in self.CHRMS:
            chrm_dat = joblib.load(dnase_folder_path + '/dnase.chr' + str(i) + '.packbit.joblib', mmap_mode='r')
            self.dnase_data_raw.append(chrm_dat['features'])
            self.dnase_labels_raw.append(chrm_dat['labels'])
            self.dnase_chrm_range[i] = (num_examples, num_examples+chrm_dat['features'].shape[0])
            num_examples += chrm_dat['features'].shape[0]            
        
        self.dnase_data = np.memmap(os.path.join(dnase_folder_path, 'dnase_seq'), dtype=self.dnase_data_raw[0].dtype, mode='w+', shape=(num_examples,1000,1))
        self.dnase_labels = np.memmap(os.path.join(dnase_folder_path, 'dnase_labels'), dtype=self.dnase_labels_raw[0].dtype, mode='w+', shape=(num_examples, self.dnase_labels_raw[0].shape[1]))
        
        num_examples = 0
        for i,x,y in zip(self.CHRMS, self.dnase_data_raw, self.dnase_labels_raw):
            self.dnase_data[num_examples:num_examples+x.shape[0]] = x
            self.dnase_labels[num_examples:num_examples+x.shape[0]] = y
            num_examples += x.shape[0]
            print('DATA ITERATOR ::: LOADED CHROMOSOME ' + str(i))
            

    def load_rna_quant(self, rna_quants_file_path):
        """
        Loads RNA quant information, an array of size (num_cell_types, num_genes). Ignores metadata. Assumes order
        of cell_types matches the order in self.dnase_labels

        Updates:
            - self.rna_quants: array of size (num_cell_types, num_genes)
        """

        self.rna_quants = joblib.load(rna_quants_file_path)['rna_quants']


    def prepare_blocks(self):
        """
        Prepares lists of chromosomes and cell types for train, validation, test sets

        Updates:
            - self.train_chrms:             set of chromosomes for training data
            - self.train_cell_types:        set of cell types for training data
            - self.validation_chrms:        set of chromosomes for validation data
            - self.validation_cell_types:   set of cell types for validation data
            - self.test_chrms:              set of chromosomes for test data
            - self.test_cell_types:         set of cell types for test data

            - self.train_indices:   numpy array of shape (num_train_examples,) that has indices of self.dnase_data that can
                                    be used for training
        """
        self.train_chrms = set(self.CHRMS)
        self.train_cell_types = set(range(self.num_cell_types))
        self.validation_chrms = set(self.CHRMS)
        self.validation_cell_types = set(range(self.num_cell_types))
        self.test_chrms = set(self.CHRMS)
        self.test_cell_types = set(range(self.num_cell_types))
        
        if self.hold_out == 'chromosomes':
            self.train_chrms -= set(self.validation_list+self.test_list)
            self.validation_chrms = set(self.validation_list)
            self.test_chrms = set(self.test_list)

        elif self.hold_out == 'cell_types':
            if self.validation_list or self.test_list:
                self.train_cell_types -= set(self.validation_list+self.test_list)
                self.validation_cell_types = set(self.validation_list)
                self.test_cell_types = set(self.test_list)
                
            # both validation and test list empty => special code for moving all cell
            # types to test - used when using cell types that don't have dnase 
            # TODO: unify with labels_avail arg
            else:   
                self.train_cell_types = set()
                self.validation_cell_types = set()

        else:
            print('DATA ITERATOR ::: HOLD_OUT NOT DEFINED')
            exit(0)

        train_indices, validation_indices, test_indices = [], [], []
        for chrm in self.train_chrms:
            train_indices.append(np.arange(*self.dnase_chrm_range[chrm]))
        for chrm in self.validation_chrms:
            validation_indices.append(np.arange(*self.dnase_chrm_range[chrm]))
        for chrm in self.test_chrms:
            test_indices.append(np.arange(*self.dnase_chrm_range[chrm]))

        self.train_indices = np.hstack(train_indices).astype(np.int32)
        self.validation_indices = np.hstack(validation_indices).astype(np.int32)
        self.test_indices = np.hstack(test_indices).astype(np.int32)

        self.num_train_examples = self.train_indices.size * len(self.train_cell_types)
        self.num_validation_examples = self.validation_indices.size * len(self.validation_cell_types)
        self.num_test_examples = self.test_indices.size * len(self.test_cell_types)

        print('DATA ITERATOR ::: NUM TRAINING EXAMPLES   : ' + str(self.train_indices.size) + ' x ' + str(len(self.train_cell_types)) + ' = ' +
              str(self.num_train_examples))
        print('DATA ITERATOR ::: NUM VALIDATION EXAMPLES : ' + str(self.validation_indices.size) + ' x ' + str(len(self.validation_cell_types)) + ' = ' +
              str(self.num_validation_examples))
        print('DATA ITERATOR ::: NUM TESTING EXAMPLES    : ' + str(self.test_indices.size) + ' x ' + str(len(self.test_cell_types)) + ' = ' +
              str(self.num_test_examples))

        # preparing mean of locus across training cell types
        avail_cell_types = np.array(sorted(self.train_cell_types)).astype(np.int8)
        if len(avail_cell_types)==0:
            if self.return_locus_mean:
                avail_cell_types = range(123)    # TODO: hack for imputation - use all training cells types (for mean models)       
            else:
                avail_cell_types = range(1)    # TODO: hack, may break things, assumes don't care about labels
        
        self.dnase_label_mean = np.mean(np.unpackbits(self.dnase_labels, axis=-1)[:, avail_cell_types] , axis=1)

        # below for training only (prep for balancing classes)
        if self.balance_classes_train:
            avail_inds, avail_cell_types, _ = self._map_type('train')
            avail_cell_types = np.array(sorted(avail_cell_types)).astype(np.int8)
            labels_avail = self.dnase_labels[avail_inds]
            labels_avail = np.array(np.unpackbits(labels_avail, axis=-1), dtype=np.int8)
            labels_avail = labels_avail[:, avail_cell_types]

            pos_seq_positions, pos_ct_positions = np.where(labels_avail>0)  # tuple of arrays: (seq_indices, cell_type_indices)
            self.pos_seq_inds = avail_inds[pos_seq_positions].astype(np.int32)
            self.pos_cell_inds = avail_cell_types[pos_ct_positions].astype(np.int32)

    def _map_type(self, type):
        if type == 'train':
            return self.train_indices, self.train_cell_types, self.train_chrms
        elif type == 'validation':
            return self.validation_indices, self.validation_cell_types, self.validation_chrms
        elif type == 'test':
            return self.test_indices, self.test_cell_types, self.test_chrms
        else:
            raise Exception("type is one of [train, validation, test]")

    def _fetch_labels_batch(self, seq_inds, ctypes):
        """
        Fetches a batch of labels of batch_size |seq_inds|=|ctypes|. The ith label has sequence seq_inds[i] and
        cell type ctypes[i]. Unpacks labels. See _fetch_seq_genexp_labels_batch for more details.
        """

        assert(seq_inds.size==ctypes.size)

        labels = self.dnase_labels[seq_inds]
        labels = np.array(np.unpackbits(labels, axis=-1), dtype=np.int64)  # unpacking the bitpacked labels
        # labels = [:, :self.num_cell_types]            # removing extra channels (only num_cell_types) - not required
        labels = labels[np.arange(seq_inds.size), ctypes]

        return labels
        
    def _fetch_seq_genexp_batch(self, seq_inds, ctypes):
        assert(seq_inds.size==ctypes.size)
        
        sequences = self.dnase_data[seq_inds]
        sequences = np.array(np.unpackbits(sequences, axis=-1), dtype=np.float32)   # unpacking the bitpacked labels
        sequences = sequences[:, :, :4]                 # removing extra channels (only 4)

        gene_exp = self.rna_quants[ctypes]
        
        return sequences, gene_exp
        
    def _fetch_seq_genexp_labels_batch(self, seq_inds, ctypes):
        """
        Fetches a batch of examples of batch_size |seq_inds|=|ctypes|. The ith example has sequence seq_inds[i] and
        cell type ctypes[i]. Unpacks labels and one-hot encoded sequences.

        Input : seq_inds, ctypes
            - seq_inds:    numpy array of dim (batch_size,), all elements correspond to a valid sequence index
            - ctypes:      numpy array of dim (batch_size,), all elements correspond to a valid cell type index

        Returns: (sequence, gene_exp, label)
            - sequence:    numpy array of dim (batch_size, 1000, 4), one hot encoded with 4 channels (ATCG)
            - gene_exp:    numpy array of dim (batch_size, num_genes)
            - label:       binary numpy array of dim (batch_size,)
        """
        assert(seq_inds.size==ctypes.size)

        sequences, gene_exp = self._fetch_seq_genexp_batch(seq_inds, ctypes)
        labels = self._fetch_labels_batch(seq_inds, ctypes)

        return sequences, gene_exp, labels

    def sample_batch(self, batch_size, type):
        """
        Samples locus from self.{train,validation,test}_indices, samples cell types from self.{train,validation,test}_cell_types.
        Unpacks labels and one-hot encoded sequences. returns the training data.

        Input : batch_size, type

        Returns: (sequence, gene_exp, label)
            - sequence:    numpy array of dim (batch_size, 1000, 4), one hot encoded with 4 channels (ATCG)
            - gene_exp:    numpy array of dim (batch_size, num_genes)
            - label:       binary numpy array of dim (batch_size,)
        """
        avail_inds, avail_cell_types, _ = self._map_type(type)

        seq_inds = np.random.choice(avail_inds, batch_size)
        ctypes = np.random.choice(np.array(list(avail_cell_types)), batch_size)

        return self._fetch_seq_genexp_labels_batch(seq_inds, ctypes) + self.return_locus_mean*(self.dnase_label_mean[seq_inds],)

    def sample_batch_balanced_train(self, batch_size):
        pos_select = np.random.choice(len(self.pos_seq_inds), int(self.positive_prop*batch_size))
        pos_seq, pos_gene, pos_target = self._fetch_seq_genexp_labels_batch(self.pos_seq_inds[pos_select], self.pos_cell_inds[pos_select])

        if self.return_locus_mean:
            neg_seq, neg_gene, neg_target, neg_label_means = self.sample_batch(batch_size, 'train')        # not all are negative! => sampled some extra
            neg_inds = [i for i in range(len(neg_target)) if neg_target[i]==0]
            neg_inds = np.random.choice(neg_inds, int((1- self.positive_prop)*batch_size))

            return np.vstack((pos_seq, neg_seq[neg_inds])), np.vstack((pos_gene, neg_gene[neg_inds])), np.concatenate((pos_target, neg_target[neg_inds])), np.concatenate((self.dnase_label_mean[self.pos_seq_inds[pos_select]], neg_label_means[neg_inds]))

        else:
            neg_seq, neg_gene, neg_target = self.sample_batch(batch_size, 'train')        # not all are negative! => sampled some extra
            neg_inds = [i for i in range(len(neg_target)) if neg_target[i]==0]
            neg_inds = np.random.choice(neg_inds, int((1 - self.positive_prop)*batch_size))

            return np.vstack((pos_seq, neg_seq[neg_inds])), np.vstack((pos_gene, neg_gene[neg_inds])), np.concatenate((pos_target, neg_target[neg_inds]))

    def sample_train_batch(self, batch_size):
        if self.balance_classes_train:
            return self.sample_batch_balanced_train(batch_size)
        return self.sample_batch(batch_size, type='train')

    def sample_validation_batch(self, batch_size):
        return self.sample_batch(batch_size, type='validation')


    def eval_generator(self, batch_size, type):
        """
        Generator for val/test data. Goes cell_type by cell_type. Subsamples by factor self.eval_subsample
        Unpacks labels and one-hot encoded sequences. Returns the data without labels.

        Input : batch_size, type

        Yields: (sequence, gene_exp) or (sequence, gene_exp, locus_mean)
            - sequence:    numpy array of dim (batch_size, 1000, 4), one hot encoded with 4 channels (ATCG)
            - gene_exp:    numpy array of dim (batch_size, num_genes)
            - locus_mean:  numpy array of dim (batch_sie)   [if self.return_locus_mean]

        Usage:
            preds = []
            for seq_batch, gene_exp_batch in di.eval_generator(BATCH_SIZE, TYPE):
                preds.append(model.predict(seq_batch, gene_exp_batch))
        """
        print("eval_subsample is {}".format(self.eval_subsample))
        avail_inds, avail_cell_types, avail_chrms = self._map_type(type)

        for ctype in avail_cell_types:
            for chrm in avail_chrms:
                start, end = self.dnase_chrm_range[chrm]
                reduced_inds = range(start, end, self.eval_subsample)   # subsampling for quicker evaluation
                for i in range(0, len(reduced_inds), batch_size):
                    seq_inds = reduced_inds[i:i+batch_size]   # 1 < len(seq_inds) <= batch_size [edge cases]
                    yield self._fetch_seq_genexp_batch(np.array(seq_inds), np.array([ctype]*len(seq_inds))) + self.return_locus_mean*(self.dnase_label_mean[seq_inds],)


    def validation_eval_generator(self, batch_size):
        return self.eval_generator(batch_size, type='validation')

    def test_eval_generator(self, batch_size):
        return self.eval_generator(batch_size, type='test')


    def sample_batch_basset(self, batch_size, type):
        """
        Samples locus from self.{train,validation,test}_indices.
        Unpacks labels and one-hot encoded sequences. returns the training data for vanilla basset (all cell types/restricted cell types).

        Input : batch_size, type

        Returns: (sequence, label)
            - sequence:    numpy array of dim (batch_size, 1000, 4), one hot encoded with 4 channels (ATCG)
            - label:       binary numpy array of dim (batch_size, num_cell_types)
        """
        if self.hold_out == 'chromosomes':
            avail_inds, avail_cell_types, _ = self._map_type(type)
        elif self.hold_out == 'cell_types':
            avail_inds, avail_cell_types = self.train_indices, self.train_cell_types    # always use train cell types for basset_vanilla!
            # TODO: better split for val/test
            thresh = int(self.train_indices.size*0.1)
            if type=='validation':
                avail_inds = avail_inds[:thresh]
            elif type=='train':
                avail_inds = avail_inds[thresh:]


        seq_inds = np.random.choice(avail_inds, batch_size)

        sequences = self.dnase_data[seq_inds]
        sequences = np.array(np.unpackbits(sequences, axis=-1), dtype=np.float32)   # unpacking the bitpacked labels
        sequences = sequences[:, :, :4]                 # removing extra channels (only 4)

        labels = self.dnase_labels[seq_inds]
        labels = np.array(np.unpackbits(labels, axis=-1), dtype=np.int64)  # unpacking the bitpacked labels
        labels = labels[:, list(sorted(avail_cell_types))]                 # avail_cell_types should be all cell types (don't hold out cell types!)

        return sequences, labels

    def sample_train_batch_basset(self, batch_size):
        return self.sample_batch_basset(batch_size, type='train')

    def sample_validation_batch_basset(self, batch_size):
        return self.sample_batch_basset(batch_size, type='validation')


    def _auprc_auc_pr(self, y_test, y_pred):
        auprc = sklearn.metrics.average_precision_score(y_test, y_pred)
        auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
        prec, rec, _ = sklearn.metrics.precision_recall_curve(y_test, y_pred)

        return auprc, auc, prec, rec

    def _update_stats(self, auprcs, aucs, precs, recs, legends, auprc, auc, prec, rec, legend):
        # need to subsample to bring size of prec and rec ~ 1000
        auprcs.append(auprc)
        aucs.append(auc)
        precs.append(prec[0:len(prec):max(1,int(len(prec)/1000))])
        recs.append(rec[0:len(rec):max(1,int(len(rec)/1000))])
        legends.append(legend)

    def _save_plot(self, precs, recs, legends, foldername, filename, title):
        data = []

        for i in range(len(precs)):
            color = '%06X' % randint(0, 0xFFFFFF)
            trace = go.Scatter(x=recs[i], y=precs[i], mode='lines', line=dict(color=color, width=2), name=legends[i])
            data.append(trace)

        layout = go.Layout(title=title, xaxis=dict(title='Recall'), yaxis=dict(title='Precision'))
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=foldername + '/' + filename + '.html')
    
    def populate_ctype_chr_pred_dict(self, avail_cell_types, avail_chrms, preds, ret_labels=True):
        labels = []
        ctype_chr_pred_dict = {}    # ctype_chr_pred_dict[ctype][chrm] will have keys 'preds' and 'labels'
        j = 0
        for ctype in avail_cell_types:
            ctype_chr_pred_dict[ctype] = {}
            for chrm in avail_chrms:
                ctype_chr_pred_dict[ctype][chrm] = {}
                start, end = self.dnase_chrm_range[chrm]
                reduced_inds = range(start, end, self.eval_subsample)   # subsampling for quicker evaluation (has to match eval_generator order)

                ctype_chr_pred_dict[ctype][chrm]['preds'] = preds[j:j+len(reduced_inds)]
                if ret_labels:
                    ctype_chr_pred_dict[ctype][chrm]['labels'] = self._fetch_labels_batch(np.array(reduced_inds), np.array([ctype]*len(reduced_inds)))
                    labels.append(self._fetch_labels_batch(np.array(reduced_inds), np.array([ctype]*len(reduced_inds))))
                j += len(reduced_inds)
                
        if ret_labels:
            labels = np.hstack(labels)
            return ctype_chr_pred_dict, labels
        else:
            return ctype_chr_pred_dict,	# comma is important for evaluate_model
    
    def evaluate_model(self, preds, type, foldername, report_filename='report'):
        """
        Performs a complete evaluation of the model predictions.

        Input : preds, type, foldername
            - preds:        a list where each element 0<=preds[i]<=1 gives the probability of chromatin accessibility, in the
                            same order as that of the corresponding eval_generator
            - type:         'validation' or 'test'
            - foldername:   folder where all analysis files will be stored
            - report_filename:   name of text file where consolidated report is stored
        """

        _, avail_cell_types, avail_chrms = self._map_type(type)
        
        f = open(foldername + '/' + report_filename + '.txt', 'w')
        f.write('-'*17 + '\n' + 'EVALUATION REPORT\n' + '-'*17 + '\n')
        f.write('TYPE: ' + type + '\n')
        f.write('HELD OUT : ' + self.hold_out + '\n')
        f.write(type + ' chromosomes : ' + str([avail_chrms]) + '\n')
        f.write(type + ' cell types  : ' + str([avail_cell_types]) + '\n\n')

        preds = np.array(preds)
        
        # ctype_chr_pred_dict[ctype][chrm] will have keys 'preds' and 'labels'
        ctype_chr_pred_dict, labels = self.populate_ctype_chr_pred_dict(avail_cell_types, avail_chrms, preds, ret_labels=True)      
        assert(labels.size==preds.size)

        os.system('mkdir ' + foldername + '/plots')

        # overall stats
        auprc, auc, prec, rec = self._auprc_auc_pr(labels, preds)
        f.write('OVERALL AUPRC = {0:0.03f}, AUC = {1:0.03f}'.format(auprc,auc) + '\n')

        if self.hold_out == 'cell_types':
            ctype_auprcs, ctype_aucs, ctype_precs, ctype_recs, ctype_legends = [], [], [], [], []
            self._update_stats(ctype_auprcs, ctype_aucs, ctype_precs, ctype_recs, ctype_legends, auprc, auc, prec, rec, 'OVERALL AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc,auc))

            for ctype in avail_cell_types:
                # ctype stats
                auprcs, aucs, precs, recs, legends = [], [], [], [], []

                auprc, auc, prec, rec = self._auprc_auc_pr(np.hstack([ctype_chr_pred_dict[ctype][x]['labels'] for x in ctype_chr_pred_dict[ctype]]), np.hstack([ctype_chr_pred_dict[ctype][x]['preds'] for x in ctype_chr_pred_dict[ctype]]))
                self._update_stats(auprcs, aucs, precs, recs, legends, auprc, auc, prec, rec, 'CELL TYPE : ' + str(ctype) + ' AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc,auc))
                self._update_stats(ctype_auprcs, ctype_aucs, ctype_precs, ctype_recs, ctype_legends, auprc, auc, prec, rec, 'CELL TYPE : ' + str(ctype) + ' AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc,auc))
                f.write('\nCELL TYPE : ' + str(ctype) + ' - AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc,auc) + '\n')

                for chrm in avail_chrms:
                    # ctype x chrm stats
                    auprc, auc, prec, rec = self._auprc_auc_pr(ctype_chr_pred_dict[ctype][chrm]['labels'], ctype_chr_pred_dict[ctype][chrm]['preds'])
                    self._update_stats(auprcs, aucs, precs, recs, legends, auprc, auc, prec, rec, 'CHRM ' + str(chrm) + ', AUPRC : {0:0.03f} , AUC : {1:0.03f}'.format(auprc,auc))
                    f.write('CELL TYPE : ' + str(ctype) + ' CHRM : ' + str(chrm) + ' - AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc,auc) + '\n')

                self._save_plot(precs, recs, legends, foldername + '/plots', 'ctype_' + str(ctype), 'CELL TYPE : ' + str(ctype))
            self._save_plot(ctype_precs, ctype_recs, ctype_legends, foldername + '/plots', 'summary', 'SUMMARY')

        elif self.hold_out == 'chromosomes':
            chrm_auprcs, chrm_aucs, chrm_precs, chrm_recs, chrm_legends = [], [], [], [], []
            self._update_stats(chrm_auprcs,chrm_aucs, chrm_precs,chrm_recs, chrm_legends, auprc, auc, prec, rec, 'OVERALL AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc, auc))

            for chrm in avail_chrms:
                #chrm stats
                auprcs, aucs, precs, recs, legends = [], [], [], [], []

                auprc, auc, prec, rec = self._auprc_auc_pr(np.hstack([ctype_chr_pred_dict[x][chrm]['labels'] for x in ctype_chr_pred_dict]), np.hstack([ctype_chr_pred_dict[x][chrm]['preds'] for x in ctype_chr_pred_dict]))
                self._update_stats(auprcs, aucs, precs, recs, legends, auprc, auc, prec, rec, 'CHRM : ' + str(chrm) + ' AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc,auc))
                self._update_stats(chrm_auprcs,chrm_aucs,chrm_precs,chrm_recs, chrm_legends, auprc, auc, prec, rec, 'CHRM : ' + str(chrm) + ' AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc,auc))
                f.write('\nCHRM : ' + str(chrm) + ' - AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc,auc) + '\n')

                for ctype in avail_cell_types:
                    # ctype x chrm stats
                    auprc, auc, prec, rec = self._auprc_auc_pr(ctype_chr_pred_dict[ctype][chrm]['labels'], ctype_chr_pred_dict[ctype][chrm]['preds'])
                    self._update_stats(auprcs, aucs, precs, recs, legends, auprc, auc, prec, rec, 'CELL TYPE ' + str(ctype) + ', AUPRC : {0:0.03f} , AUC : {1:0.03f}'.format(auprc,auc))
                    f.write('CELL TYPE : ' + str(ctype) + ' CHRM : ' + str(chrm) + ' - AUPRC = {0:0.03f} , AUC = {1:0.03f}'.format(auprc,auc) + '\n')

                self._save_plot(precs, recs, legends, foldername + '/plots', 'chrm_' + str(chrm), 'CHRM : ' + str(chrm))
            self._save_plot(chrm_precs, chrm_recs, chrm_legends, foldername + '/plots', 'summary', 'SUMMARY')

        f.close()
