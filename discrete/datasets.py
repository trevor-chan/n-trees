import sys
sys.path.append('../n-trees/')
sys.path.append('../utils/')
import numpy as np
import torch
from fast_generator import generate_forest
from torch.utils.data import Dataset
import utils
import tqdm



class ForestDataset(Dataset):
    def __init__(self, d, n, temp=1, maxiter=1000000, size=10000):
        self.d = d
        self.n = n
        self.temp = temp
        self.maxiter = maxiter
        
        self.forests = self.__generate_data(size)
        self.size = size
        self.data_std = np.std(self.forests[0][0])
        self.data_p = np.where(self.forests[0][0] == 1)[0].shape[0] / (self.d * self.n)
        
    def __generate_data(self, size):
        forests = []
        for i in tqdm.tqdm(range(size)):
            forests.append(generate_forest(self.d, self.n, self.temp, self.maxiter))
        forests = np.stack(forests).astype(np.uint8)
        return forests
        
    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        dictionary = self.__dict__.copy()
        dictionary.pop('forests')
        utils.dict_to_json(dict(dictionary), filepath + '/source.json')
        utils.save_array(self.forests, filepath + '/data.npz')
    
    @classmethod
    def load(cls, source_path):
        dataset = cls()
        assert os.path.exists(source_path), 'provided path does not exist'
        assert os.path.exists(source_path+'/source.json'), 'missing source file'
        assert os.path.exists(source_path+'/data.npz') or os.path.exists(source_path+'/data.npy'), 'missing mask file'
        
        source = utils.json_to_dict(source_path+'/source.json')
        
        dataset.d = source['d']
        dataset.n = source['n']
        dataset.temp = source['temp']
        dataset.maxiter = source['maxiter']
        dataset.size = source['size']
        dataset.data_std = source['data_std']
        dataset.data_p = source['data_p']
        dataset.forests = utils.load_array(source_path+'/data')
        return dataset

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


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1