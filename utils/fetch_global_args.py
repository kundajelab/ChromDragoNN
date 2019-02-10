import argparse
import os

DATA_DIR = '/data/'
ALL_CHROMOSOMES = list(range(1,23)) + ['X', 'Y']

def common_args(parser):
    parser.add_argument('--momentum', type=float, default=0.98, help='Momentum for Optimizer, if applicable')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout Rate')

    parser.add_argument('--cuda', type=int, default=1, help='GPU availibility (1 if available, else 0)')
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='Total number of epochs')
    parser.add_argument('-bpe', '--batches_per_epoch', type=int, default=4000, help='Number of batches per train epoch')
    parser.add_argument('-bpte', '--batches_per_test_epoch', type=int, default=500, help='Number of batches per validation epoch')
    parser.add_argument('-sc', '--schedule', type=list, default=[], help='Adjust learning rate schedule')

    parser.add_argument('-rb', '--resume_from_best', type=int, default=0, help='Resume from best model in checkpoint directory')
    parser.add_argument('-ev', '--evaluate', type=int, default=0, help='Evaluate model on test set by default, else pass -evonval 1')
    parser.add_argument('-evonval', '--eval_on_validation', type=int, default=0, help='Evaluate model on validation set (default is test)')
    parser.add_argument('-evmode', '--eval_mode', type=str, default='report', help='Evaluation mode: report [generates reports and plots] or save_preds [saves predictions with labels, if labels_avail is True]')
    parser.add_argument('-rfn', '--report_filename', type=str, default='report', help='Name of report file, if eval_mode == report')   
    parser.add_argument('-la', '--labels_avail', type=int, default=1, help='Default 1, set to 0 if labels are not available (i.e. evaluating on new cell types)')

    parser.add_argument('-ho', '--hold_out', type=str, default='cell_types', help='Hold out: chromosomes or cell_types')
    parser.add_argument('-vl', '--validation_list', nargs = '*', type=int, default=[1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
                         help='elements in validation_set; if \'chromosomes\' are held out, this list will \
                               contain numbers from list(range(1,23)) + [\'X\',\'Y\']; if \'cell_types\' then \
                               numbers from list(range(num_cell_types))')
    parser.add_argument('-tl', '--test_list', nargs = '*', type=int, default=[15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
                         help='elements in test_set; if \'chromosomes\' are held out, this list will \
                               contain numbers from list(range(1,23)) + [\'X\',\'Y\']; if \'cell_types\' then \
                               numbers from list(range(num_cell_types))')

    parser.add_argument('--dnase', type=str, default=os.path.join(DATA_DIR, 'packbited'), help='Path to DNase data folder')
    parser.add_argument('--rna_quants', type=str, default=os.path.join(DATA_DIR, 'rna_quants_1630tf.joblib'), help='Path to RNA Quants joblib file')
    parser.add_argument('-chrms', '--chromosomes', type=list, default=ALL_CHROMOSOMES, help='Chromosomes to load from data iterator (use default unless debugging!)')
    parser.add_argument('-ntct', '--num_total_cell_types', type=int, default=123, help='Number of cell types in dataset (use default unless changing dataset!)')

    parser.add_argument('-cp', '--checkpoint', type=str, default=None, required=True, help='Folder path to save models')


def stage1_global_argparser():
    parser = argparse.ArgumentParser()
    common_args(parser)

    parser.add_argument('--lr', '--learning_rate', type=float, default=0.002, help='Learning Rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='Batch Size')

    return parser

def stage2_global_argparser():
    parser = argparse.ArgumentParser()
    common_args(parser)

    parser.add_argument('--lr', '--learning_rate', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch Size')

    parser.add_argument('-fr', '--freeze_pretrained_model', type=int, default=1, help='Freeze Stage1 model parameters')
    parser.add_argument('-pp', '--positive_proportion', type=float, default=0.25, help='Ratio of positive to negative samples during training')
    parser.add_argument('-pcw', '--pos_class_weight', type=float, default=1.0, help='Weight of positive example relative to negative in loss function')
    parser.add_argument('-s1m', '--stage1_pretrained_model_path', type=str, default=None, required=True, help='Path of pretrained Stage 1 model folder')
    parser.add_argument('-s1f', '--stage1_file', type=str, default=None, required=True, help='Path of Stage 1 model python file')
    parser.add_argument('-ng', '--num_genes', type=int, default=1630, help='Number of genes in rna-exp vector')
    parser.add_argument('-wm', '--with_mean', type=int, default=0, help='Use mean accessibility as input feature')

    return parser

def orig_data_global_argparser():
    parser = argparse.ArgumentParser()
    common_args(parser)

    parser.add_argument('--lr', '--learning_rate', type=float, default=0.002, help='Learning Rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='Batch Size')

    print('ARGPARSER ::: Make sure you set batches_per_test_epoch correctly')
    parser.add_argument('--data_path', type=str, default=os.path.join(DATA_DIR,'basset_orig_data.joblib'), help='Path to processed original Basset data joblib file')

    return parser
