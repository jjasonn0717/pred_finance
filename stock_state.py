import numpy as np
import time
import sys
import csv
from yahoo_finance import Share
import  matplotlib.pyplot as plt
import random
import argparse
from scipy.fftpack import fft

DAY_DELAY = 1
DATA_LEN = 7
ACTIONS = 2
BIDS = 1.

class Stock_state:
    def __init__(self, csv_path, bids, action_num, DAY_RANGE=15, start_y=None, end_y=None, Adjust=None, min_len=None):
        self.bids = bids
        self.DAY_RANGE = DAY_RANGE
        self.adjust = Adjust
        data_file = open(csv_path, 'r')
        reader = csv.reader(data_file)
        self.dates = []
        self.d_Close = []
        self.Close = []
        self.daily_data = np.empty((0, DATA_LEN), dtype='float')
        self.header = np.array(reader.next()) # Dates High Low Open Close d_Close RSI9 RSI15 MA5 MA20 MA60 d_CO d_HL Adi_Close Volume VA/D d_VA/D %R8 %R21 DIF DEM d_MA5-20
        
        ## d_Close, d_MA5-20, d_VA/D, RSI9, RSI15, DIF, DEM ##
        ids = np.array([5, 21, 16, 6, 7, 19, 20])
        for line in reader:
            if not start_y == None:
                year = line[0].split('-')[0]
                if year < start_y or year > end_y:
                    continue
            self.dates.append(line[0])
            self.d_Close.append(float(line[5]))
            self.Close.append(float(line[4]))
            data = np.array(line[1:], dtype='float')
            data = [data[ids-1]]
            self.daily_data = np.append(self.daily_data, data, axis=0)
        ## if no data in given years range, raise an error ##
        if len(self.dates) == 0:
            raise ValueError("no complete data from "+start_y+"~"+end_y)
        ## data normalization ##
        if self.adjust == None:
            self.adjust = {"subs":np.array([], dtype='float'), "divs":np.array([], dtype='float')}
            sub = np.mean(self.daily_data[:,0:3], axis=0)
            div = np.std(self.daily_data[:,0:3], axis=0)
            self.daily_data[:,0:3] = (self.daily_data[:,0:3] - sub) / div ## normalize
            self.adjust["subs"] = np.concatenate((self.adjust["subs"], sub))
            self.adjust["divs"] = np.concatenate((self.adjust["divs"], div))

            self.adjust["subs"] = np.concatenate((self.adjust["subs"], [0., 0.]))
            self.adjust["divs"] = np.concatenate((self.adjust["divs"], [1., 1.]))
            
            sub = [np.mean(self.daily_data[:,5:7])]*2
            div = [np.std(self.daily_data[:,5:7])]*2
            self.daily_data[:,5:7] = (self.daily_data[:,5:7] - sub) / div ## min-max scaling
            self.adjust["subs"] = np.concatenate((self.adjust["subs"], sub))
            self.adjust["divs"] = np.concatenate((self.adjust["divs"], div))
            
            assert len(self.adjust["subs"]) == DATA_LEN and len(self.adjust["divs"]) == DATA_LEN
        else:
            self.daily_data = (self.daily_data - self.adjust["subs"]) / self.adjust["divs"]
            
        self.data_size = len(self.dates)
        assert self.daily_data.shape == (self.data_size, DATA_LEN)
        print 'Load data', self.dates[0], 'to', self.dates[-1], 'from ' + self.header[0] + ',', self.data_size, 'days in total'
        self.header = self.header[ids]
        data_file.close()

        ## for RNN ##
        self.randomList = np.arange(self.data_size-self.DAY_RANGE + 1)
        random.shuffle(self.randomList)
        self.next_start = 0
        self.next_start_nonrand = 0

        ## for money ##
        self.revenue = 0.

        ## for random episode ##
        self.min_len = min_len
        if self.min_len == None:
            self.epi_len = self.data_size
        else:
            self.epi_len = random.randrange(self.min_len, self.data_size+1)
        self.current = random.randrange(self.data_size - self.epi_len + 1)
        self.epi_start = self.current

    def SetMinLen(self, min_len):
        self.min_len = min_len
        self.epi_len = random.randrange(self.min_len, self.data_size+1)
        self.current = random.randrange(self.data_size - self.epi_len + 1)
        self.epi_start = self.current

    def GetSize(self):
        return self.data_size

    def GetAdjust(self):
        return self.adjust

    def GetClose(self):
        return self.Close

    def Invest_step(self, action):
        next_day = self.daily_data[self.current]
        if self.current + DAY_DELAY >= self.epi_start + self.epi_len:
            '''p_close = self.daily_data[self.data_size-1][3]
            p_open = self.daily_data[self.data_size-1][2]
            d_price = p_close - p_open '''
            d_price = self.d_Close[self.epi_len-1]
        else:
            '''p_close = self.daily_data[self.current+DAY_DELAY][3]
            p_open = self.daily_data[self.current+DAY_DELAY][2]
            d_price = p_close - p_open'''
            d_price = self.d_Close[self.current+DAY_DELAY]
        date = self.dates[self.current]
        '''if d_price > p_close*0.005:
            p_array = np.array([-1, 1, -1], dtype='float')
        elif d_price >= p_open*(-0.005) and d_price <= p_open*0.005:
            p_array = np.array([1, -1, -1], dtype='float')
        else:
            p_array = np.array([-1, -1, 1], dtype='float')
        act_correct = np.dot(p_array, np.asarray(action, dtype='float'))
        reward = act_correct*self.bids*max(np.abs(d_price), 0.01)'''

        if d_price >= 0:
            p_array = np.array([1, -1], dtype='float') ## ACTION_NUM = 2, 0:buy ; 1:sell
        else:
            p_array = np.array([-1, 1], dtype='float')
        act_correct = np.dot(p_array, np.asarray(action, dtype='float'))
        #reward = act_correct*self.bids*max(np.abs(d_price), 0.001)
        reward = act_correct*self.bids*np.abs(d_price)
        self.revenue += reward

        revenue = self.revenue
        if (self.current - self.epi_start) == self.epi_len - 1:
            if not self.min_len == None:
                self.epi_len = random.randrange(self.min_len, self.data_size+1)
            self.current = random.randrange(self.data_size - self.epi_len + 1)
            self.epi_start = self.current
            self.revenue = 0.
            terminate = True
        else:
            self.current += 1
            terminate = False
        return date, next_day, reward, terminate, revenue

    def GetEpiLen(self):
        return self.epi_len

    def NextRNNBatch(self, batch_size):
        if self.next_start + batch_size < (self.data_size - self.DAY_RANGE + 1):
            real_batch_size = batch_size
        else:
            real_batch_size = (self.data_size - self.DAY_RANGE) - self.next_start + 1
        
        batch_data = np.empty((0, self.DAY_RANGE, DATA_LEN), dtype='float')
        batch_label = np.empty((0, 2), dtype='float')
        batch_r_t = []
        for b_num in xrange(real_batch_size):
            day_range_data = np.empty((0, DATA_LEN), dtype='float')
            location = self.randomList[self.next_start + b_num]
            for day_num in xrange(self.DAY_RANGE):
                single_day = self.daily_data[location + day_num]
                day_range_data = np.append(day_range_data, [single_day], axis=0)
            assert day_range_data.shape == (self.DAY_RANGE, DATA_LEN)
            batch_data = np.append(batch_data, [day_range_data], axis=0)
            
            label_loca = (location + self.DAY_RANGE + DAY_DELAY) if (location + self.DAY_RANGE + DAY_DELAY) < self.data_size else self.data_size - 1
            r_t = self.d_Close[label_loca]
            if r_t >= 0:
                label = [1, 0] ## ACTION_NUM = 2, 0:buy ; 1:sell
            else:
                label = [0, 1]
            batch_label = np.append(batch_label, [label], axis=0)
            batch_r_t.append(abs(r_t))
        assert batch_data.shape == (real_batch_size, self.DAY_RANGE, DATA_LEN) and batch_label.shape == (real_batch_size, 2)

        if self.next_start + real_batch_size < self.data_size - self.DAY_RANGE + 1:
            self.next_start = self.next_start + real_batch_size
            terminal = False
        else:
            self.next_start = 0
            random.shuffle(self.randomList)
            terminal = True

        return batch_data, batch_label, batch_r_t, terminal

    def TestRNNBatch(self, batch_size):
        if self.next_start_nonrand + batch_size < (self.data_size - self.DAY_RANGE + 1):
            real_batch_size = batch_size
        else:
            real_batch_size = (self.data_size - self.DAY_RANGE) - self.next_start_nonrand + 1
        
        batch_data = np.empty((0, self.DAY_RANGE, DATA_LEN), dtype='float')
        batch_label = np.empty((0, 2), dtype='float')
        batch_r_t = []
        for b_num in xrange(real_batch_size):
            day_range_data = np.empty((0, DATA_LEN), dtype='float')
            location = self.next_start_nonrand + b_num
            for day_num in xrange(self.DAY_RANGE):
                single_day = self.daily_data[location + day_num]
                day_range_data = np.append(day_range_data, [single_day], axis=0)
            assert day_range_data.shape == (self.DAY_RANGE, DATA_LEN)
            batch_data = np.append(batch_data, [day_range_data], axis=0)
            
            label_loca = (location + self.DAY_RANGE + DAY_DELAY) if (location + self.DAY_RANGE + DAY_DELAY) < self.data_size else self.data_size - 1
            r_t = self.d_Close[label_loca]
            if r_t >= 0:
                label = [1, 0] ## ACTION_NUM = 2, 0:buy ; 1:sell
            else:
                label = [0, 1]
            batch_label = np.append(batch_label, [label], axis=0)
            batch_r_t.append(abs(r_t))
        assert batch_data.shape == (real_batch_size, self.DAY_RANGE, DATA_LEN) and batch_label.shape == (real_batch_size, 2)

        if self.next_start_nonrand + real_batch_size < self.data_size - self.DAY_RANGE + 1:
            self.next_start_nonrand = self.next_start_nonrand + real_batch_size
            terminal = False
        else:
            self.next_start_nonrand = 0
            terminal = True

        return batch_data, batch_label, batch_r_t, terminal

    def show_data(self):
        ## show choice ##
        print "choices:",
        for i, name in enumerate(self.header):
            print "["+str(i)+' '+name+"]",
        print '\n'

        plt.ion()
        while True:
            ## get request column to draw ##
            while True:
                column = raw_input(">>> request column: ")
                if column == "q":
                    sys.exit(0)
                try:
                    column = int(column)
                    if column < 0 or column >= len(self.header):
                        print "Invalid column:", column, " Request column should be a positive int!"
                        column = None

                except:
                        print "Invalid column:", column, " Request column should be a positive int!"
                        column = None
                if not column == None:
                    break
            print "Chosen column:", self.header[column], '\n'
            plt.plot(self.daily_data[:,column], 'r-')#, data[:,column-1], 'ro')
            plt.title(self.header[column])
            plt.show()

    def RandGuess(self):
        mul = [1., -1.]
        rand_act = np.array([random.sample(mul, 1) for i in xrange(len(self.dates))]).reshape([-1])
        rev = np.dot(self.d_Close, rand_act)
        return rev
        


def getYear(mode):
    while True:
        years = raw_input(">>> Year range of "+mode+" set(START END): ")
        years = years.split()
        try:
            years = [int(y) for y in years]
            if not len(years) == 2:
                print "should be two numbers seperated by space!!"
                continue
        except:
            print "should be two numbers seperated by space!!"
            continue 
        break
    return [str(y) for y in years]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="draw input stock data")
    parser.add_argument("comp", help="the source of the data to draw")
    args = parser.parse_args()

    train_range = getYear("training")
    test_range = getYear("testing")
    try:
        market = Stock_state('./stock_data/'+args.comp+'.csv',
                             bids=1.,
                             action_num=ACTIONS,
                             start_y=train_range[0],
                             end_y=train_range[1])
        adj = market.GetAdjust()
        test_market = Stock_state('./stock_data/'+args.comp+'.csv',
                                  bids=1.,
                                  action_num=ACTIONS,
                                  start_y=test_range[0],
                                  end_y=test_range[1],
                                  Adjust=adj)
    except:
        print "There is no data of " + args.comp + " in folder stock_data!"
        sys.exit(0)

    while True:
        Type = raw_input(">>> train, test or rand? ")
        if Type == "q":
            sys.exit(0)
        elif Type == "train" or Type == "test" or Type == "rand":
            break
        else:
            print "Invalid mode "+Type+"!\n"
    if Type == "train":
        market.show_data()
    elif Type == "test":
        test_market.show_data()
    else:
        print test_market.RandGuess()

