import sys
import random
import numpy as np
import math
import argparse
from blist import sortedlist, blist


def find_first(l, val):
    for i, ele in enumerate(l):
        if ele == val:
            return i
    return None


class myQueue:
    def __init__(self, capacity, batch_size, alpha=0.6, beta_0=0.4):
        self.capacity       = capacity
        self.alpha          = alpha
        self.beta           = beta_0
        self.batch_size     = batch_size
        
        self.batch_segs     = []
        self.IS_weights     = None
        
        self.TDErr_list     = sortedlist([])
        self.Tran_list      = blist([])
        self.time_list      = np.array([], dtype='int')

        ## pre-calc segment boundary ##
        rank_P = np.array([1./p for p in np.arange(1, self.capacity+1, dtype='float')])
        rank_P = np.power(rank_P, self.alpha) / np.sum(np.power(rank_P, self.alpha))
        # segment boundary #
        seg_p = 1./self.batch_size
        seg_num = 1
        seg_start = 0
        acc_p = 0.
        for i, p in enumerate(rank_P):
            acc_p += p
            if acc_p >= seg_p*seg_num:
                self.batch_segs.append((seg_start, i))
                seg_start = i + 1
                seg_num += 1
        if len(self.batch_segs) == self.batch_size - 1:
            self.batch_segs.append((seg_start, self.capacity - 1))
        assert len(self.batch_segs) == self.batch_size
        # set Important Sampling weights #
        self.setIS()

    def setIS(self):
        self.IS_weights = np.concatenate([[1./(e-s+1)]*(e-s+1) for s, e in self.batch_segs])
        self.IS_weights = np.power((1./self.IS_weights)*(1./self.capacity), self.beta)
        self.IS_weights = self.IS_weights / np.max(self.IS_weights)
        assert len(self.IS_weights) == self.capacity

    def append(self, new_tran, TD_err):
        if len(self.TDErr_list) == self.capacity:
            self.delOldest()
        if not len(self.TDErr_list) == 0:
            self.time_list += 1 # increase time step for every transition
        TD_err = (-1.)*abs(TD_err) # make maximum mininmum
        idx = self.TDErr_list.bisect_left(TD_err)
        self.TDErr_list.add(TD_err)
        self.Tran_list.insert(idx, new_tran)
        self.time_list = np.insert(self.time_list, idx, 0)
        assert len(self.TDErr_list) == len(self.time_list) and len(self.TDErr_list) == len(self.Tran_list)

    def delOldest(self):
        idx = np.argmax(self.time_list)
        del self.Tran_list[idx]
        del self.TDErr_list[idx]
        self.time_list = np.delete(self.time_list, idx)

    def GetBatch(self):
        batch = []
        batch_time = []
        batch_IS = []
        if len(self.TDErr_list) == self.capacity:
            for k in xrange(self.batch_size):
                begin = self.batch_segs[k][0]
                end = self.batch_segs[k][1] + 1
                sample_idx = random.randrange(begin, end)
                batch.append(self.Tran_list[sample_idx])
                batch_time.append(self.time_list[sample_idx])
                batch_IS.append(self.IS_weights[sample_idx])
        else:
            if len(self.TDErr_list) <= self.batch_size:
                batch = list(self.Tran_list)
                batch_time = list(self.time_list)
                batch_IS = [1.]*len(self.TDErr_list)
            else:
                begin = 0
                end = len(self.TDErr_list)
                sample = random.sample(np.arange(len(self.TDErr_list)), self.batch_size)
                for sample_idx in sample:
                    batch.append(self.Tran_list[sample_idx])
                    batch_time.append(self.time_list[sample_idx])
                    batch_IS.append(1.)
            
        assert len(batch) == self.batch_size or len(batch) == len(self.TDErr_list)
        
        return batch, batch_time, batch_IS


    def UpdateTDErr(self, batch_time, batch_TDErr):
        assert len(batch_time) == self.batch_size and len(batch_TDErr) == self.batch_size
        for i, time in enumerate(batch_time):
            idx = find_first(self.time_list, time)
            ## save temp ##
            tran = self.Tran_list[idx]
            ## delete old ##
            del self.TDErr_list[idx]
            del self.Tran_list[idx]
            self.time_list = np.delete(self.time_list, idx)
            ## add new ##
            new_TDErr = (-1.)*abs(batch_TDErr[i])
            new_idx = self.TDErr_list.bisect_left(new_TDErr)
            self.TDErr_list.add(new_TDErr)
            self.Tran_list.insert(new_idx, tran)
            self.time_list = np.insert(self.time_list, new_idx, time)
        assert len(self.TDErr_list) == len(self.time_list) and len(self.TDErr_list) == len(self.Tran_list)


        
