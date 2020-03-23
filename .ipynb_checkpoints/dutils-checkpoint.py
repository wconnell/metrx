import copy
import numpy as np
import pandas as pd
from tcga_datasets import TCGA
from collections import OrderedDict

class Experiment():
    """
    Defines an experimental class hierarchy object.
    """
    def __init__(self, samples_dir, hierarchy, cases, min_samples):
        self.samples_dir = samples_dir
        self.hierarchy = hierarchy
        self.samples = self.load_data(self.samples_dir, self.hierarchy, min_samples)
        self.cases = self.samples[cases].unique()
        self.meta_dict = {key:val for key,val in enumerate(self.samples['meta'].cat.categories.values)}
        
    def load_data(self, samples_dir, hierarchy, min_samples):
        assert isinstance(hierarchy, OrderedDict), "Argument of wrong type."
        samples = pd.read_csv(samples_dir, sep="\t")
        samples['data'] = [val[1] for i,val in samples['File Name'].str.split(".").items()]
        # downsample data
        for key,val in hierarchy.items():
            samples = samples[samples[key].isin(val)]
        # unique meta classes
        samples['meta'] = samples[list(hierarchy.keys())].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
        # filter meta classes
        meta_counts = samples['meta'].value_counts()
        meta_keep = meta_counts[meta_counts > min_samples].index
        samples = samples[samples['meta'].isin(meta_keep)]
        # generate class categories
        samples['meta'] = samples['meta'].astype('category')
        return samples
    
    def holdout(self, holdout):
        self.holdout = holdout
        self.holdout_samples = self.samples[self.samples['meta'].isin(holdout)]
        self.samples = self.samples[~self.samples['meta'].isin(holdout)]
    
def train_test_split_case(samples, cases, test_size=0.20):
    """
    Splits dataframe into random train and test sets for Siamese network evaluation.
    
    N matched samples for cases are distributed by test_size and each case conserved in one dataset arm.
    
    """
    # total test size, round to even number for subsetting test
    n_test = int(len(samples) * test_size)
    if n_test % 2 != 0: n_test = n_test - 1
    # subset samples with a single matched pair
    ids = samples.groupby([cases]).size()
    # take test_size samples from each case with multiple paired samples
    test_cases = np.array([])
    for i in range(ids.min(), ids.max()+1):
        nlets = ids[ids == i].index.values
        nlets = np.random.choice(nlets, size=int(len(nlets)*test_size), replace=False)
        test_cases = np.concatenate([test_cases, nlets])
    return samples[~samples[cases].isin(test_cases)].reset_index(drop=True).sample(frac=1),\
           samples[samples[cases].isin(test_cases)].reset_index(drop=True).sample(frac=1)

