
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
import importlib.util

ALL_CHROMOSOMES = list(range(1,23)) + ['X', 'Y']
STAGE1_CHECKPOINTS = '/scratch/users/surag/cs273b/models/sepval/stage1/'
STAGE2_CHECKPOINTS = '/scratch/users/surag/cs273b/models/sepval/stage2/'
INF = np.inf

def flatten_dict_of_dicts(data, key = 'preds'):
    assert(type(data) == dict and type(data[next(iter(data.keys()))]) == dict)
    flatten_cell = lambda x:  np.hstack((y[key] for y in data[x].values()))
    matrix = np.vstack(tuple(map(flatten_cell, data.keys())))
    return matrix


def import_net_from_file(filename):
    spec = importlib.util.spec_from_file_location("module", filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Net


class dotdict(dict):
    # https://stackoverflow.com/questions/42272335/how-to-make-a-class-which-has-getattr-properly-pickable
   
    __slots__ = ()
       
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value
 



def calibration_test(preds, labels, show = False):
    bins = np.linspace(0, 1, 22)
    digitized = np.digitize(preds, bins)
    mapper = lambda x: (preds[np.where(digitized == x)].mean(),
                        labels[np.where(digitized == x)].mean())
    avg_preds, avg_labels = zip(*list(map(mapper, range(1,bins.shape[0]))))

    plt.figure()
    plt.plot(bins, bins, label = 'accurate')
    plt.plot(avg_preds, avg_labels, 'r+-.', label = 'predictions')
    plt.title('Calibration Plot')
    plt.xlabel('Mean Predicted Value For Each Bin')
    plt.ylabel('Fraction of True Positive Cases')
    if show:
        plt.show()

    plt.savefig('../Plots/calibration.png')



