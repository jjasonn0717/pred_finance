#!/usr/bin/env python
##--------------------------------------##
## double DQN,                          ##
## Target Freezing,                     ##
## multi-layer LSTM, CNN with inception ##
##--------------------------------------##
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import sys
import random
import numpy as np
from collections import deque
import stock_state
import math
import argparse
import matplotlib.pyplot as plt

'''ACTIONS = 3                                  # number of valid actions # 0: do nothing, 1: buy, 2: sell
ACTIONS_NAME = ['DO NOTHING', 'BUY', 'SELL']    # action name'''
ACTIONS = 2                                     # number of valid actions # 0: buy, 1: sell
ACTIONS_NAME = ['BUY', 'SELL']                  # action name
GAMMA = 0.99                                    # decay rate of past observations
OBSERVE = 2                                     # episode to observe before training
EXPLORE = 1000.                                 # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001                          # final value of epsilon
INITIAL_EPSILON = 0.1                           # starting value of epsilon
REPLAY_MEMORY = 8000                            # number of previous transitions to remember
TARGET_UPDATE_FREQ = 100                        # step interval to update target network
BATCH = 32                                      # size of minibatch

DAYS_RANGE = 15                                 # columns of input data
INPUT_DIM = 7                                   # rows of input data

HIGH_THRE = 0.4                                 # threshold of reward for calc accuracy

SUMMARY_FOLDER = "./summary_ver3/"              # folder for storing summary
SAVED_PATH = "saved_networks_ver3"              # folder for storing models


def getYear(mode):
    while True:
        years = raw_input(">>> Year range of "+mode+" set(START END): ")
        years = years.split()
        try:
            years = [int(y) for y in years]
            if not len(years) == 2:
                print("should be two numbers seperated by space!!")
                continue
        except:
            print("should be two numbers seperated by space!!")
            continue 
        break
    return [str(y) for y in years]



class MultiMarket:
    def __init__(self, market_path):
        self.market_num = 0
        self.markets = []
        self.test_markets = []
        self.adjust = []
        self.comp = []
        train_range = getYear("training")
        test_range = getYear("testing")
        file = open(market_path, 'r')
        for line in file:
            try:
                self.markets.append(stock_state.Stock_state('./stock_data/'+line[:-1]+'.csv',
                                                            bids=1.,
                                                            action_num=ACTIONS,
                                                            DAY_RANGE=DAYS_RANGE,
                                                            start_y=train_range[0],
                                                            end_y=train_range[1]))
                adj = self.markets[-1].GetAdjust()
                self.test_markets.append(stock_state.Stock_state('./stock_data/'+line[:-1]+'.csv',
                                                                 bids=1.,
                                                                 action_num=ACTIONS,
                                                                 DAY_RANGE=DAYS_RANGE,
                                                                 start_y=test_range[0],
                                                                 end_y=test_range[1],
                                                                 Adjust=adj))
                #self.markets[-1].SetMinLen(self.test_markets[-1].GetSize())
                self.adjust.append(adj)
                self.market_num += 1
                self.comp.append(line[:-1])
            except IOError:
                raise Exception("stock data of comp. " + line[:-1] + "not found!")
            except ValueError as e:
                raise Exception("error: "+str(e))
        file.close()

    def NextTrainMkt(self):
        idx = random.randrange(self.market_num)
        print("Next Train Market:", self.comp[idx], "  epi_len: ", self.markets[idx].GetEpiLen())
        return self.markets[idx], self.comp[idx]

    def NextTestMkt(self):
        idx = random.randrange(self.market_num)
        print("Next Test Market:", self.comp[idx])
        return self.test_markets[idx], self.comp[idx]

    def InferMkt(self, idx=0):
        print("Inference Market:", self.comp[idx])
        return self.test_markets[idx], self.comp[idx]



class DQN_module:
    def __init__(self):
        
        ## open up market manager ##
        self.MarketMgr = MultiMarket("./stock_data/market.txt")
        self.Market         = None
        self.test_Market    = None

        ## store the previous observations in replay memory ##
        self.D              = deque()
        
        ## init train session ##
        self.sess           = tf.InteractiveSession()
        ## init record session ##
        self.g_record       = tf.Graph()
        self.sess_record    = tf.InteractiveSession(graph=self.g_record)
        
        self.update_freq    = TARGET_UPDATE_FREQ   # interval of updating target network
        self.update_list    = []                   # list of updating target network ops
        self.s              = None                 # input observed state
        self.target_s       = None                 # target input observed state
        self.readout        = None                 # output actions Q value
        self.target_readout = None                 # target output actions Q value
        self.prob           = None                 # dropout keep prob

        ## training ##
        self.in_action      = None
        self.targetQ        = None
        self.cost           = None
        self.train_step     = None
        self.global_step    = None

        self.merged         = None
        self.init_op        = None
        
        ## record ##
        self.close_price    = None
        self.action_Qval    = None
        self.rec_revenue    = None
        self.merged_record  = None

        ## test record ##
        self.testAcc        = None
        self.testRev        = None
        self.merged_test    = None

        self.test_comp_name = None
        
        ## build DQN model ##
        #self.createCNN()                          # CNN network
        self.createMultiRNN(2, 1024)               # LSTM network
        
        ## create saver ##
        with self.sess.graph.as_default():
            self.saver = tf.train.Saver()
        self.createOP()
        
        ## init network var ##
        self.sess.run(self.init_op)


    def weight_variable(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        fan_out = np.prod(shape[1:])
        std = math.sqrt(1.0 / (fan_in + fan_out))
        initial = tf.truncated_normal(shape, stddev = std)
        #initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        fan_out = np.prod(shape[1:])
        std = math.sqrt(1.0 / (fan_in + fan_out))
        #initial = tf.random_uniform([shape[-1]], minval=(-std), maxval=std)
        initial = tf.constant(0.01, shape=[shape[-1]])
        return tf.Variable(initial, name=name)

    def conv2d_relu(self, x, target_x, kernel, stride, name):
        with tf.name_scope(name):
            ## training ##
            W = self.weight_variable(kernel, name='w')
            b = self.bias_variable(kernel, name='b')
            activation = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME', name='conv2d') + b, name='relu')
            
            ## target ##
            target_W = tf.Variable(W.initialized_value(), trainable=False, name='target_w')
            target_b = tf.Variable(b.initialized_value(), trainable=False, name='target_b')
            target_activation = tf.nn.relu(tf.nn.conv2d(target_x, target_W, strides=[1, stride, stride, 1], padding='SAME', name='target_conv2d') + b, name='target_relu')

            tf.histogram_summary(name+'w', W)
            tf.histogram_summary(name+'b', b)

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name):
            tf.add_to_collection("L2_VARIABLES", var)

        self.update_list.append(target_W.assign(W))
        self.update_list.append(target_b.assign(b))
        
        return [activation, target_activation]

    def FC_layer(self, x, target_x, kernel, name, activate):
        with tf.name_scope(name):
            ## training ##
            W = self.weight_variable(kernel, name='w')
            b = self.bias_variable(kernel, name='b')
            z_out = tf.matmul(x, W) + b
            
            ## target ##
            target_W = tf.Variable(W.initialized_value(), trainable=False, name='target_w')
            target_b = tf.Variable(b.initialized_value(), trainable=False, name='target_b')
            target_z_out = tf.matmul(target_x, target_W) + target_b

            tf.histogram_summary(name+'w', W)
            tf.histogram_summary(name+'b', b)

            if activate:
                activation = tf.nn.relu(z_out, name='relu')
                target_activation = tf.nn.relu(target_z_out, name='target_relu')
            else:
                activation = z_out
                target_activation = target_z_out

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name):
            tf.add_to_collection("L2_VARIABLES", var)

        self.update_list.append(target_W.assign(W))
        self.update_list.append(target_b.assign(b))
        
        return [activation, target_activation]

    def max_pool_2x2(self, x, name):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", name=name+'pool')


    def createMultiRNN(self, n_layer, n_hidden):
       
        with self.sess.graph.as_default():
            self.prob = tf.placeholder("float", name="keep_prob")
            # input #
            with tf.name_scope('input'):
                self.s = tf.placeholder('float', shape=[None, INPUT_DIM, DAYS_RANGE], name='input_state')
                input_trans = tf.transpose(self.s, [2, 0, 1]) # [DAYS_RANGE, None, INPUT_DIM]
                input_reshape = tf.reshape(input_trans, [-1, INPUT_DIM])
                input_list = tf.split(0, DAYS_RANGE, input_reshape) # split to DAY_RANGE element

            with tf.name_scope('tg_input'):
                self.target_s = tf.placeholder('float', shape=[None, INPUT_DIM, DAYS_RANGE], name='input_state')
                tg_input_trans = tf.transpose(self.target_s, [2, 0, 1]) # [DAYS_RANGE, None, INPUT_DIM]
                tg_input_reshape = tf.reshape(tg_input_trans, [-1, INPUT_DIM])
                tg_input_list = tf.split(0, DAYS_RANGE, tg_input_reshape) # split to DAY_RANGE element

            # multi LSTM #
            lstm_cell = rnn_cell.LSTMCell(n_hidden, use_peepholes=True, forget_bias=1.0, state_is_tuple=True)
            lstm_drop = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.prob)
            lstm_stack = rnn_cell.MultiRNNCell([lstm_drop] * n_layer, state_is_tuple=True)

            tg_lstm_cell = rnn_cell.LSTMCell(n_hidden, use_peepholes=True, forget_bias=1.0, state_is_tuple=True)
            tg_lstm_drop = rnn_cell.DropoutWrapper(tg_lstm_cell, output_keep_prob=self.prob)
            tg_lstm_stack = rnn_cell.MultiRNNCell([tg_lstm_drop] * n_layer, state_is_tuple=True)

            lstm_output, hidden_states = rnn.rnn(lstm_stack,
                                                 input_list,
                                                 dtype='float',
                                                 scope='LSTMStack') # out: [timestep, batch, hidden], state: [cell, 2(for c, h), batch, hidden]
            tg_lstm_output, tg_hidden_states = rnn.rnn(tg_lstm_stack, tg_input_list, dtype='float', scope='tg_LSTMStack')
            
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="LSTMStack"):
                tf.add_to_collection("L2_VARIABLES", var)
            
            h_fc1 = self.FC_layer(lstm_output[-1], tg_lstm_output[-1], [n_hidden, 1024], name='h_fc1', activate=True)
            h_fc2 = self.FC_layer(h_fc1[0], h_fc1[1], [1024, ACTIONS], name='h_fc2', activate=False)

            key = tf.GraphKeys.TRAINABLE_VARIABLES
            update_pair = zip(tf.get_collection(key, scope="LSTMStack"), tf.get_collection(key, scope="tg_LSTMStack"))
            for var, tg_var in update_pair:
                self.update_list.append(tg_var.assign(var))

            # readout layer
            self.readout = h_fc2[0]
            self.target_readout = h_fc2[1]


    def createCNN(self):
       
        with self.sess.graph.as_default():
            # input layer #
            with tf.name_scope('input'):
                self.s = tf.placeholder("float", [None, INPUT_DIM, DAYS_RANGE], name='input_state')
                s_4d = tf.reshape(self.s, [-1, INPUT_DIM, DAYS_RANGE, 1])
            
            with tf.name_scope('target_input'):
                self.target_s = tf.placeholder("float", [None, INPUT_DIM, DAYS_RANGE], name='target_input_state')
                target_s_4d = tf.reshape(self.target_s, [-1, INPUT_DIM, DAYS_RANGE, 1])

            h_conv1_fx = self.conv2d_relu(s_4d, target_s_4d, [INPUT_DIM, 3, 1, 32], 1, name='h_conv1_fx')
            h_conv1_2x = self.conv2d_relu(s_4d, target_s_4d, [2, 5, 1, 32], 1, name='h_conv1_2x')
            h_conv1_1x = self.conv2d_relu(s_4d, target_s_4d, [1, 5, 1, 32], 1, name='h_conv1_1x')
            
            h_conv1_fx_inc = self.conv2d_relu(h_conv1_fx[0], h_conv1_fx[1], [1, 1, 32, 12], 1, name='h_conv1_fx_inc')
            h_conv1_2x_inc = self.conv2d_relu(h_conv1_2x[0], h_conv1_2x[1], [1, 1, 32, 10], 1, name='h_conv1_2x_inc')
            h_conv1_1x_inc = self.conv2d_relu(h_conv1_1x[0], h_conv1_1x[1], [1, 1, 32, 10], 1, name='h_conv1_1x_inc')
            
            h_conv1_dcon = tf.concat(3, [h_conv1_1x_inc[0], h_conv1_2x_inc[0], h_conv1_fx_inc[0]])
            target_h_conv1_dcon = tf.concat(3, [h_conv1_1x_inc[1], h_conv1_2x_inc[1], h_conv1_fx_inc[1]])
            
            h_conv2 = self.conv2d_relu(h_conv1_dcon, target_h_conv1_dcon, [INPUT_DIM, 3, 32, 64], 1, name='h_conv2')
            h_conv3 = self.conv2d_relu(h_conv2[0], h_conv2[1], [INPUT_DIM, 2, 64, 64], 1, name='h_conv3')
            
            conv3_flat = tf.reshape(h_conv3[0], [-1, INPUT_DIM*DAYS_RANGE*64], name='flatten')
            target_conv3_flat = tf.reshape(h_conv3[1], [-1, INPUT_DIM*DAYS_RANGE*64], name='target_flatten')
            
            h_fc1 = self.FC_layer(conv3_flat, target_conv3_flat, [INPUT_DIM*DAYS_RANGE*64, 1024], name='h_fc1', activate=True)
            h_fc2 = self.FC_layer(h_fc1[0], h_fc1[1], [1024, ACTIONS], name='h_fc2', activate=False)

            # readout layer #
            self.readout = h_fc2[0]
            self.target_readout = h_fc2[1]


    def createOP(self):
        ## cost function and training method ##
        with self.sess.graph.as_default():
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            with tf.name_scope('train'):
                self.in_action = tf.placeholder("float", [None, ACTIONS], 'input_action')
                self.targetQ = tf.placeholder("float", [None], 'input_Q')
                
                l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.get_collection("L2_VARIABLES")]) # l2 loss
                print([var.name for var in tf.get_collection("L2_VARIABLES")])

                readout_actionQ = tf.reduce_sum(tf.mul(self.readout, self.in_action), reduction_indices=1) # Q value of chosen action
                self.cost = tf.reduce_mean(tf.square(self.targetQ - readout_actionQ))# + 0.001*l2_loss # total cost
                
                # training op #
                start_l_rate = 0.0005
                decay_step = 100000
                decay_rate = 0.5
                learning_rate = tf.train.exponential_decay(start_l_rate, self.global_step, decay_step, decay_rate, staircase=False)
                grad_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_step = tf.contrib.layers.optimize_loss(loss=self.cost, 
                                                             global_step=self.global_step, 
                                                             learning_rate=0.0005, 
                                                             optimizer=grad_op, 
                                                             clip_gradients=1)
                tf.scalar_summary('learning_rate', learning_rate)
                tf.scalar_summary('l2_loss', l2_loss)
        
        ## training op ##
        with self.sess.graph.as_default():
            self.merged = tf.merge_all_summaries()
            self.init_op = tf.initialize_all_variables()
        
        ## record op ##
        with self.g_record.as_default():

            with tf.name_scope('record'):
                self.close_price = tf.placeholder('float', name='close_price')
                self.action_Qval = tf.placeholder('float', [ACTIONS], 'action_Qval')
                self.rec_revenue = tf.placeholder('float', name='revenue')

                close_price_summ = tf.scalar_summary('close_price', self.close_price)
                buy_Qval_summ = tf.scalar_summary('buy_Qval', self.action_Qval[0])
                sell_Qval_summ = tf.scalar_summary('sell_Qval', self.action_Qval[1])
                rec_revenue_summ = tf.scalar_summary('revenue', self.rec_revenue)
            
            self.merged_record = tf.merge_summary([close_price_summ, buy_Qval_summ, sell_Qval_summ, rec_revenue_summ])
        
        ## test acc op ##
        with self.g_record.as_default():
            
            with tf.name_scope('accuracy'):
                self.testAcc = tf.placeholder('float', name='accuracy')
                self.testRev = tf.placeholder('float', name='test_revenue')

                testAcc_summ = tf.scalar_summary('accuracy', self.testAcc)
                testRev_summ = tf.scalar_summary('test_revenue', self.testRev)
            self.merged_test = tf.merge_summary([testAcc_summ, testRev_summ])


    def TestAcc(self, step):
        
        if step % 10000 == 0 or self.test_Market == None:
            self.test_Market, self.test_comp_name = self.MarketMgr.NextTestMkt()
        # stack the first DAYS_RANGE state by doing nothing to get the first input #
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        date_t, x_t, r_0, terminal_0, rev_0 = self.test_Market.Invest_step(do_nothing)
        s_t = x_t.reshape(INPUT_DIM, 1)
        print('Testing '+self.test_comp_name+' Start at Date: ' + date_t + '   epi_len: ' + str(self.test_Market.GetEpiLen()))
        for i in xrange(DAYS_RANGE-1):
            date_t, x_t, r_0, terminal_0, rev_0 = self.test_Market.Invest_step(do_nothing)
            s_t = np.append(s_t, x_t.reshape(INPUT_DIM, 1), axis=1)
        assert s_t.shape[1] == DAYS_RANGE
        
        ## calc ACC ##
        valid_days = 0.
        RT_high = 0.
        RT_low = 0.
        corr_RT_high = 0.
        corr_RT_low = 0.
        correct_pred = 0.
        total_days = 0.
        while True:
            readout_t = self.sess.run(self.readout, feed_dict={self.s : [s_t], self.prob : 1.})[0]
            a_t = np.zeros([ACTIONS])
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            date_t1, x_t1, r_t, terminal_t, rev_t = self.test_Market.Invest_step(a_t)
            x_t1 = np.reshape(x_t1, (INPUT_DIM, 1))
            s_t1 = np.append(s_t[:,1:], x_t1, axis = 1)
            if r_t > 0:
                correct_pred += 1
                if r_t > HIGH_THRE:
                    corr_RT_high += 1.
                else:
                    corr_RT_low += 1.
            if not r_t == 0:
                valid_days += 1
                if abs(r_t) > HIGH_THRE:
                    RT_high += 1.
                else:
                    RT_low += 1.
            total_days += 1
            '''if total_days % 10 == 0 or terminal_t:
                print("TIMESTEP", total_days,
                      "/ Date", date_t1, "/ ACTION", ACTIONS_NAME[action_index], "/ REWARD", r_t,
                      "/ Q_MAX %e" % np.max(readout_t), '/ Revenue %.4f' % rev_t)'''
            if terminal_t:
                final_rev = rev_t
                acc = correct_pred / valid_days
                h_acc = corr_RT_high / RT_high if RT_high > 0 else 0.
                l_acc = corr_RT_low / RT_low if RT_low > 0 else 0.
                print('Testing '+self.test_comp_name+' End at Date: ' + date_t1 + 
                      '  ACC: %.4f' % acc, 
                      ' H_ACC: %.4f' % h_acc, 
                      ' L_ACC: %.4f' % l_acc, 
                      '  REV: %.4f' % final_rev)
                break
            s_t = s_t1
        return acc, final_rev


    def Inference(self):
        ## loading networks ##
        checkpoint = tf.train.get_checkpoint_state(SAVED_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
            self.sess_record.close()
            self.sess.close()
            sys.exit(0)
        
        while True:
            idx = raw_input(">>>market idx: ")
            try:
                idx = int(idx)
                self.test_Market, comp_name = self.MarketMgr.InferMkt(idx)
                break
            except:
                print("Invalid idx "+str(idx))
        # get the first DAYS_RANGE state by doing nothing to get the first input stack #
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        date_t, x_t, r_0, terminal_0, rev_0 = self.test_Market.Invest_step(do_nothing)
        s_t = x_t.reshape(INPUT_DIM, 1)
        print('Testing '+comp_name+' Start at Date: ' + date_t)
        for i in xrange(DAYS_RANGE-1):
            date_t, x_t, r_0, terminal_0, rev_0 = self.test_Market.Invest_step(do_nothing)
            s_t = np.append(s_t, x_t.reshape(INPUT_DIM, 1), axis=1)
        assert s_t.shape[1] == DAYS_RANGE
       
        plt.ion()
        color_map = ['g-', 'r-']

        ## calc ACC ##
        valid_days = 0.
        RT_high = 0.
        RT_low = 0.
        corr_RT_high = 0.
        corr_RT_low = 0.
        correct_pred = 0.
        total_days = 0.
        Close_p = self.test_Market.GetClose()[DAYS_RANGE-1:-1]
        full_corr_or_not = []
        while True:
            readout_t = self.sess.run(self.readout, feed_dict={self.s : [s_t], self.prob : 1.})[0]
            a_t = np.zeros([ACTIONS])
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            date_t1, x_t1, r_t, terminal_t, rev_t = self.test_Market.Invest_step(a_t)
            x_t1 = np.reshape(x_t1, (INPUT_DIM, 1))
            s_t1 = np.append(s_t[:,1:], x_t1, axis = 1)
            if r_t > 0:
                correct_pred += 1
                if r_t > HIGH_THRE:
                    corr_RT_high += 1.
                else:
                    corr_RT_low += 1.
                full_corr_or_not.append(1)
            else:
                full_corr_or_not.append(0)
            if not r_t == 0:
                valid_days += 1
                if abs(r_t) > HIGH_THRE:
                    RT_high += 1.
                else:
                    RT_low += 1.
            total_days += 1
            if total_days % 50 == 0 or terminal_t:
                print("STEP", total_days,
                      "/ ", date_t1, "/ ", ACTIONS_NAME[action_index], "/ REWARD", r_t,
                      "/ Q_MAX %e" % np.max(readout_t), '/ Revenue %.4f' % rev_t)
            if terminal_t:
                final_rev = rev_t
                acc = correct_pred / valid_days
                h_acc = corr_RT_high / RT_high if RT_high > 0 else 0.
                l_acc = corr_RT_low / RT_low if RT_low > 0 else 0.
                print('Testing '+comp_name+' End at Date: ' + date_t1 + 
                      '  ACC: %.4f' % acc, 
                      ' H_ACC: %.4f' % h_acc, 
                      ' L_ACC: %.4f' % l_acc, 
                      ' (H/L): ' + str(RT_high) + '/' + str(RT_low), 
                      '  REV: %.4f' % final_rev)
                break
            s_t = s_t1
        self.sess_record.close()
        self.sess.close()

        ## draw action-close_price figure ##
        assert len(full_corr_or_not) == len(Close_p)
        for i in xrange(len(full_corr_or_not)-1):
            plt.plot([i, i+1], Close_p[i:i+2], color_map[int(full_corr_or_not[i])])
        plt.show()
        raw_input("press to continue...")


    def trainNetwork(self):
        
        ## create summaries folder ##
        if tf.gfile.Exists(SUMMARY_FOLDER+'train'):
            tf.gfile.DeleteRecursively(SUMMARY_FOLDER+'train')
        tf.gfile.MakeDirs(SUMMARY_FOLDER+'train')
        train_writer = tf.train.SummaryWriter(SUMMARY_FOLDER+'train', self.sess.graph)

        if tf.gfile.Exists(SUMMARY_FOLDER+'record'):
            tf.gfile.DeleteRecursively(SUMMARY_FOLDER+'record')
        tf.gfile.MakeDirs(SUMMARY_FOLDER+'record')
        record_writer = tf.train.SummaryWriter(SUMMARY_FOLDER+'record', self.sess_record.graph)
        
        
        ## loading networks ##
        checkpoint = tf.train.get_checkpoint_state(SAVED_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        ## start training ##
        epsilon = INITIAL_EPSILON
        t = 0
        state = "observe"
        episode = 0
        market_fin = False
        while True:
            self.Market, comp_name = self.MarketMgr.NextTrainMkt()
            # get the first state by doing nothing and duplcate DAYS_RANGE time to get the first input stack #
            do_nothing = np.zeros(ACTIONS)
            do_nothing[0] = 1
            date_t, x_t, r_0, terminal_0, revenue_0 = self.Market.Invest_step(do_nothing)
            s_t = np.stack((x_t,)*DAYS_RANGE, axis=1)
            
            correct_train_pred = 0.
            valid_train = 0.
            corr_RT_high = 0.
            RT_high = 0.
            corr_RT_low = 0.
            RT_low = 0.
            total_train = 0.
            while "flappy bird" != "angry bird":
                
                # choose an action epsilon greedily #
                readout_t = self.sess.run(self.readout, feed_dict={self.s : [s_t], self.prob : 1.})[0]
                
                a_t = np.zeros([ACTIONS])
                action_index = 0
                if random.random() <= epsilon:
                    '''if not state == "observe":
                        print("----------Random Action----------")'''
                    action_index = random.randrange(ACTIONS)
                    a_t[random.randrange(ACTIONS)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1

                ## scale down epsilon ##
                if epsilon > FINAL_EPSILON and (not state == "observe"):
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                # run the selected action and observe next state and reward #
                date_t1, x_t1, r_t, terminal_t, revenue_t = self.Market.Invest_step(a_t)
               
                if terminal_t == True:
                    market_fin = True

                # record invest summary #
                if not state == 'observe' and (t+1) % 50 == 0:
                    record_summary = self.sess_record.run(self.merged_record, 
                                                          feed_dict={self.close_price : s_t[2,-1], self.action_Qval : readout_t, self.rec_revenue : revenue_t})
                    record_writer.add_summary(record_summary, t+1)
                
                # store the transition in memory #
                x_t1 = np.reshape(x_t1, (INPUT_DIM, 1))
                s_t1 = np.append(s_t[:,1:], x_t1, axis = 1)

                self.D.append((s_t, a_t, r_t, s_t1, terminal_t))
                if len(self.D) > REPLAY_MEMORY:
                    self.D.popleft()

                # only train if done observing #
                loss = 0.
                if not state == "observe":
                    # sample a minibatch to train on #
                    minibatch = random.sample(self.D, BATCH)

                    # get the batch variables #
                    s_j_batch = [d[0] for d in minibatch]
                    a_batch = [d[1] for d in minibatch]
                    r_batch = [d[2] for d in minibatch]
                    s_j1_batch = [d[3] for d in minibatch]

                    '''# DQN --- calc target Q #
                    y_batch = []
                    readout_j1_batch = self.sess.run(self.target_readout, feed_dict = {self.target_s : s_j1_batch, self.prob : 1.})
                    for i in xrange(0, len(minibatch)):
                        terminal = minibatch[i][4]
                        # if terminal, only equals reward
                        if terminal:
                            y_batch.append(r_batch[i])
                        else:
                            y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))'''
                    
                    # DDQN --- calc target Q #
                    y_batch = []
                    readout_j1_batch = self.sess.run(self.readout, feed_dict = {self.s : s_j1_batch, self.prob : 1.})
                    readout_j1_batch_tg = self.sess.run(self.target_readout, feed_dict = {self.target_s : s_j1_batch, self.prob : 1.})
                    for i in xrange(0, len(minibatch)):
                        terminal = minibatch[i][4]
                        # if terminal, only equals reward
                        if terminal:
                            y_batch.append(r_batch[i])
                        else:
                            max_idx = np.argmax(readout_j1_batch[i])
                            y_batch.append(r_batch[i] + GAMMA * readout_j1_batch_tg[i][max_idx])
                    
                    # perform gradient step #
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, summary, loss = self.sess.run([self.train_step, self.merged, self.cost], 
                                                     feed_dict = {self.targetQ : y_batch, self.in_action : a_batch, self.s : s_j_batch, self.prob : 0.85}, 
                                                     options=run_options, 
                                                     run_metadata=run_metadata)
                    # record training summary #
                    if (t+1) % 50 == 0 or market_fin == True:
                        train_writer.add_run_metadata(run_metadata, date_t1+'_'+str(episode))
                        train_writer.add_summary(summary, t+1)
                
                # update the old values #
                s_t = s_t1
                t += 1
                if not state == "observe":
                    total_train += 1
                    if r_t > 0:
                        correct_train_pred += 1
                        if r_t > HIGH_THRE:
                            corr_RT_high += 1
                        else:
                            corr_RT_low += 1
                    if not r_t == 0:
                        valid_train += 1
                        if abs(r_t) > HIGH_THRE:
                            RT_high += 1
                        else:
                            RT_low += 1
                    if t % self.update_freq == 0:
                        self.sess.run(self.update_list) # updating target network

                # report result #
                if t % 10 == 0 or market_fin == True:
                    if not state == "observe":
                        print("STEP", t, "/", state.ljust(7),
                              "/", date_t1, "/", ACTIONS_NAME[action_index].ljust(4), "/ REWARD", str(r_t).rjust(7),
                              "/ Q_MAX %e" % np.max(readout_t), 
                              "/ cost %.4f" % loss, 
                              '/ ACC %.4f' % (correct_train_pred / valid_train) if valid_train > 0. else 0., 
                              '/ H_ACC %.4f ' % ((corr_RT_high / RT_high) if RT_high > 0. else 0.), 
                              '/ L_ACC %.4f ' % ((corr_RT_low / RT_low) if RT_low > 0. else 0.), 
                              '/ REV %.4f' % revenue_t)
                    else:
                        if t % 100 == 0 or market_fin == True:
                            print("STEP", t, "/ ", state.ljust(7),
                                  "/", date_t1, "/", ACTIONS_NAME[action_index].ljust(4), "/ REWARD", str(r_t).rjust(7),
                                  "/ Q_MAX %e" % np.max(readout_t), '/ REV %.4f' % revenue_t)
                
                # testing and save networks every 200 steps #
                if t % 200 == 0 and (not state == "observe"):
                    accuracy, final_rev = self.TestAcc(t)
                    acc_summary = self.sess_record.run(self.merged_test, feed_dict={self.testAcc : accuracy, self.testRev : final_rev})
                    record_writer.add_summary(acc_summary, t)

                    self.saver.save(self.sess, SAVED_PATH + "/TSMC" + '-dqn', global_step=t)
                
                # check state #
                if episode < OBSERVE and state == "observe":
                    state = "observe"
                elif t >= EXPLORE and state == "explore":
                    state = "train"
                else:
                    state = state
                
                # check temination #
                if market_fin == True:
                    print('EPISODE %d' % (episode+1), 'finished', 'Final Revenue: %.4f' % revenue_t, '\n')
                    episode += 1
                    market_fin = False
                    if episode >= OBSERVE and state == "observe":
                        t = 0
                        state = "explore"
                    break

def main(mode):
    StockDQN = DQN_module()
    if mode == "train":
        StockDQN.trainNetwork()
    elif mode == "test":
        StockDQN.Inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Deep Q-Network")
    parser.add_argument("-m","--mode", help="mode of DQN, default to train", choices=("train", "test"), default="train")
    args = parser.parse_args()
    main(args.mode)
