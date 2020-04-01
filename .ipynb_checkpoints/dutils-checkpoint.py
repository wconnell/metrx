import copy
import numpy as np
import pandas as pd
from collections import OrderedDict

class Experiment():
    """
    Defines an experimental class hierarchy object.
    """
    def __init__(self, meta_data, hierarchy, index, cases, min_samples):
        self.hierarchy = hierarchy
        self.index = index
        self.meta_data = self.categorize(meta_data, self.hierarchy, min_samples)
        self.cases = self.meta_data[cases].unique()
        self.labels = self.meta_data['meta'].cat.codes.values.astype('int')
        self.labels_dict = {key:val for key,val in enumerate(self.meta_data['meta'].cat.categories.values)}
        
    def categorize(self, meta_data, hierarchy, min_samples):
        assert isinstance(hierarchy, OrderedDict), "Argument of wrong type."
        # downsample data
        for key,val in hierarchy.items():
            meta_data = meta_data[meta_data[key].isin(val)]
        # unique meta classes
        meta_data['meta'] = meta_data[list(hierarchy.keys())].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
        # filter meta classes
        counts = meta_data['meta'].value_counts()
        keep = counts[counts > min_samples].index
        meta_data = meta_data[meta_data['meta'].isin(keep)]
        # generate class categories
        meta_data['meta'] = meta_data['meta'].astype('category')
        return meta_data
    
    def holdout(self, holdout):
        self.holdout = holdout
        self.holdout_meta = self.meta_data[self.meta_data['meta'].isin(holdout)].set_index(keys=self.index, drop=True)
        self.meta_data = self.meta_data[~self.meta_data['meta'].isin(holdout)]
        
    def train_test_split(self, cases, test_size=0.20):
        """
        Splits meta data into random train and test sets for Siamese network evaluation.

        N matched samples for cases are distributed by test_size and each case conserved in one dataset arm.
        """
        # total test size, round to even number for subsetting test
        n_test = int(len(self.meta_data) * test_size)
        if n_test % 2 != 0: n_test = n_test - 1
        # subset samples with a single matched pair
        ids = self.meta_data.groupby([cases]).size()
        # take test_size samples from each case with multiple paired samples
        test_cases = np.array([])
        for i in range(ids.min(), ids.max()+1):
            nlets = ids[ids == i].index.values
            nlets = np.random.choice(nlets, size=int(len(nlets)*test_size), replace=False)
            test_cases = np.concatenate([test_cases, nlets])
        self.train_meta, self.test_meta = self.meta_data[~self.meta_data[cases].isin(test_cases)]\
                                                         .set_index(keys=self.index, drop=True)\
                                                         .sample(frac=1),\
                                          self.meta_data[self.meta_data[cases].isin(test_cases)]\
                                                         .set_index(keys=self.index, drop=True)\
                                                         .sample(frac=1)\
                        
    def get_data(self, data, subset, dtype=np.float32):
        """
        Takes raw matrix and returns appropriate train or test subset as defined by meta data.
        """
        if subset is "train":
            assert hasattr(self, "train_meta"), "Must define data split before use."
            data = data[data.index.isin(self.train_meta.index)].reindex(self.train_meta.index).astype(dtype)
            labels = self.train_meta['meta']
        elif subset is "test":
            assert hasattr(self, "test_meta"), "Must define data split before use."
            data = data[data.index.isin(self.test_meta.index)].reindex(self.test_meta.index).astype(dtype)
            labels = self.test_meta['meta']
        elif subset is "holdout":
            assert hasattr(self, "holdout_meta"), "Must define holdout before use."
            data = data[data.index.isin(self.holdout_meta.index)].reindex(self.holdout_meta.index).astype(dtype)
            labels = self.holdout_meta['meta']
        
        assert np.array_equal(data.index, labels.index), "Data indices and labels do not align."
        return data, labels.cat.codes.values.astype('int')

