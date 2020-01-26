import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd

class TCGA(Dataset):
    """
    Stores data as tensors for iterating
    """
    
    def __init__(self, root_dir, samples, train, target):
        self.root_dir = root_dir
        self.samples = samples
        self.train = train
        self.data = self.load_tcga_rna(self.root_dir, self.samples)
        #self.labels = self.samples[target].cat.codes.values.astype('int')
        self.labels = self.samples[target].to_numpy()
        
    def __getitem__(self, index):
        return torch.from_numpy(self.data.iloc[index].values).float(), self.labels[index]
        
    def __len__(self):
        return len(self.samples)
    
    def load_tcga_rna(self, root_dir, samples):
        alt_dir = os.path.join(root_dir, "https:/api.gdc.cancer.gov/data/")
        df_list = []

        for fid,fname in zip(samples['File ID'], samples['File Name']):

            if os.path.exists(os.path.join(root_dir, fid, fname)):
                df_list.append(pd.read_csv(os.path.join(root_dir, fid, fname), sep="\t", index_col=0, header=None).T)

            elif os.path.exists(os.path.join(alt_dir, fid, fname)):
                df_list.append(pd.read_csv(os.path.join(alt_dir, fid, fname), sep="\t", index_col=0, header=None).T)

            else:
                print("{} not found".format(os.path.join(fid, fname)))
                break

        df = pd.concat(df_list)
        df.index = samples['Sample ID']

        return df
        
        
class SiameseTCGA(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self,  tcga_dataset):
        self.train = tcga_dataset.train
        self.data = tcga_dataset.data
        self.labels = tcga_dataset.labels

        if self.train:
            self.train_labels = self.labels
            self.train_data = torch.from_numpy(self.data.values).float()
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.labels
            self.test_data = torch.from_numpy(self.data.values).float()
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i]]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
        
        return (img1, img2), target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
