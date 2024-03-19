import sys
sys.path.append('../n-trees/')
import numpy as np
import torch
from fast_generator import generate_forest
from torch.utils.data import Dataset


class ForestDataset(Dataset):
    def __init__(self, d, n, temp=1, maxiter=1000000, size=10000):
        self.d = d
        self.n = n
        self.temp = temp
        self.maxiter = maxiter
        
        self.forests = self.__generate_data(size)
        self.size = size
        
    def __generate_data(self, size):
        forests = []
        for i in range(size):
            forests.append(generate_forest(self.d, self.n, self.temp, self.maxiter))
        forests = np.stack(forests)
        return forests
        # trees = forests[:,0]
        # roots = forests[:,1]

    def __len__(self):
        return self.size
    
    def __augment(self, sample):
        sample = torch.tensor(sample, dtype=torch.int64)
        if np.random.randint(0,2):
            sample = torch.flip(sample, [1])
        sample = torch.rot90(sample, np.random.randint(0,4), [1, 2])
        
        roots_one_hot = torch.nn.functional.one_hot(sample[1]-1, num_classes=self.d).permute(2, 0, 1)
        trees = sample[0].to(torch.float32) * 2 - 1
        roots_one_hot = roots_one_hot.to(torch.float32)
        return trees, roots_one_hot

    def __getitem__(self, index):
        forest = self.forests[index]
        trees, roots_one_hot = self.__augment(forest)
        return trees, roots_one_hot
