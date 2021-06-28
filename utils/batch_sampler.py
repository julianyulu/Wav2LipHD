import random
from torch.utils.data.sampler import Sampler


class RandomEntryIter:
    def __init__(self, indexes, batch_size, steps_per_epoch):
        self.indexes = indexes
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.cnt = 0 

    def __iter__(self):
        return self 

    def __next__(self):
        self.cnt += 1
        if self.cnt > self.steps_per_epoch:
            raise StopIteration
        else:
            idx = random.choices(self.indexes, k = self.batch_size)
            return idx
    
class RandomEntryBatchSampler(Sampler):
    """
    A Batch Sampler that enables random entry such that 
    one can have batch_size larger than N/batch_size runs 
    per epoch (need to set steps per epoch) 
    """
    def __init__(self, n_samples, batch_size, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.indexes = list(range(n_samples))

    def __iter__(self):
        return RandomEntryIter(self.indexes, self.batch_size, self.steps_per_epoch)
        
    def __len__(self):
        return self.steps_per_epoch
