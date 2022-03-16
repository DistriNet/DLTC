import h5py
import numpy as np
import random
from random import shuffle
import pdb
import yaml
import pickle

random.seed(4)

'''
The pipeline provides ndarray rather than framework-specific formats.
'''
class Generator():
    def __init__(self, ids, h5_path, batch_size, is_test=False, channel_last=True, x=None, y=None):
        '''
        ids: id list
        is_test: if True, it is the test dataset, otherwise training dataset.
        channel_last: if True, corresponds to inputs with shape [batch, height, width, channels] (for tensorflow),
                      otherwise, [batch, channels, height, width] (for pytorch and paddlepaddle).
        x,y: if None, read from .h5 file.
        '''
        self.ids = ids
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.is_test = is_test
        self.channel_last = channel_last
        self.spe = int(np.ceil(len(self.ids)/self.batch_size)) # steps per epoch
        self.x = x
        self.y = y
        
    def epoch(self):
        x = []
        y = []
        ids = self.ids.copy()
        if not self.is_test:
            shuffle(ids)
        while ids:
            i = ids.pop()
            self.append(x, y, i)
            if len(x) == self.batch_size or not ids:
                yield self.feed(x, y)
                x = []
                y = []
        return
    
    def append(self, x, y, i):
        '''
        Dataset specific. 
        This is for (deepcorr)[http://traces.cs.umass.edu/index.php/Network/Network]
        notice that x, y are list.
        x,y: list to be feeded
        i: index
        '''
        if self.x is not None:
            x.append(self.x[i])
            y.append(self.y[i])
        else:
            with h5py.File(self.h5_path, 'r') as f:
                if self.channel_last:
                    x.append(f['data']['x'][i][...,np.newaxis])
                else:
                    x.append(f['data']['x'][i][np.newaxis,...])
                y.append(f['data']['y'][i])
        return
    
    def feed(self, x, y):
        return np.asarray(x), np.asarray(y)
        
class Dataset():
    def __init__(self, cf='config.yml', fold=-1, test_only=False, channel_last=True, h5_path=None, in_mem=True):
        '''
        cf: config.yml path
        fold: which test fol to evaluate.
        test_only: if True, only for test dataset.
        channel_last: if True, corresponds to inputs with shape (batch, height, width, channels) (for tensorflow),
                  otherwise, (batch, channels, height, width) (for pytorch and paddlepaddle).
        h5_path: if None, use default .h5 file in config.yml, otherwise, use the given path.
        in_mem: if True, read .h5 once and save x,y in memory.
        '''
        with open(cf) as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)
        self.h5_path = h5_path or self.config['data']['h5_path']
        self.channel_last = channel_last
        if in_mem:
            with h5py.File(self.h5_path, 'r') as f:
                self.x = np.asarray(f['data']['x'])
                self.y = np.asarray(f['data']['y'])
        else:
            self.x = self.y = None
            
        self.fold = fold
        
        if test_only:
            print("Only test: no cross-val")
            return
        
        crossval_file = self.config['data']['crossval_indices_path']
        self.n_fold = self.config['data']['n_fold']
        with open(crossval_file,'rb') as f:
            self.crossval_dict = pickle.load(f)
    
    @property
    def _train_ids(self):
        if self.fold >= 0:
            return self.crossval_dict[f'train_{self.fold}']
        else:
            return self.crossval_dict[f'train']
        
    @property
    def _val_ids(self):
        if self.fold >= 0:
            return self.crossval_dict[f'val_{self.fold}']
        else:
            return self.crossval_dict[f'val']
        
    @property
    def _test_ids(self):
        with h5py.File(self.h5_path, 'r') as f:
            if self.fold >= 0:
                return list(f['indices'][f'test{self.fold}'])
            else:
                return list(f['indices']['test'])
        
    @property
    def train_generator(self):
        return Generator(ids = self._train_ids, 
                         h5_path = self.h5_path, 
                         batch_size = self.config['train']['batch_size'],
                         channel_last = self.channel_last,
                         x = self.x, y = self.y)
    @property
    def val_generator(self):
        return Generator(ids = self._val_ids, 
                         h5_path = self.h5_path,
                         batch_size = self.config['train']['batch_size'],
                         channel_last = self.channel_last,
                         x = self.x, y = self.y)
    @property
    def test_generator(self):
        return Generator(ids = self._test_ids,
                         h5_path = self.h5_path,
                         batch_size = self.config['test']['batch_size'],
                         is_test = True,
                         channel_last = self.channel_last,
                         x = self.x, y = self.y)
    