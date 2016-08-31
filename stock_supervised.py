#!/usr/bin/env python
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import sys
import time
import random
import numpy as np
from collections import deque
import stock_state
import math
import argparse
import matplotlib.pyplot as plt


ACTIONS = 2 # number of valid actions # 0: buy, 1: sell
BATCH = 32 # size of minibatch

DAYS_RANGE = 15
INPUT_DIM = 7

SUMMARY_FOLDER = "./lstm_summary/"              # folder for storing summary
SAVED_PATH = "saved_lstm"                       # folder for storing models

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
        print "Next Train Market:", self.comp[idx], "  epi_len: ", self.markets[idx].GetEpiLen()
        return self.markets[idx], self.comp[idx]

    def NextTestMkt(self):
        idx = random.randrange(self.market_num)
        print "Next Test Market:", self.comp[idx]
        return self.test_markets[idx], self.comp[idx]

    def InferMkt(self, idx=0):
        print "Inference Market:", self.comp[idx]
        return self.test_markets[idx], self.comp[idx]



class MODEL:
    def __init__(self):
        
        ## open up market & test market ##
        self.MarketMgr = MultiMarket("./stock_data/market.txt")
        self.Market         = None
        self.test_Market    = None

        ## store the previous observations in replay memory ##
        self.D              = deque()
        
        ## init train session ##
        self.sess           = tf.InteractiveSession()
        self.g_record       = tf.Graph()
        self.sess_record    = tf.InteractiveSession(graph=self.g_record)
        
        self.s              = None                 # input observed state
        self.pred_action    = None                 # output actions prob
        self.prob           = None                 # dropout prob

        ## training ##
        self.corr_action    = None
        self.cost           = None
        self.train_step     = None
        self.global_step    = None

        self.rev            = None
        self.corr_or_not    = None
        self.accuracy       = None
        self.batch_rev      = None

        self.merged         = None
        self.init_op        = None

        ## testing ##
        self.testAcc        = None
        self.testRev        = None
        self.trainRev       = None
        self.merged_test    = None
        
        self.test_comp_name = None

        ## build  model ##
        self.createRNN()                           # RNN Network
        #self.createCNN()                          # CNN Network
        self.createOP()
        
        ## create saver ##
        with self.sess.graph.as_default():
            self.saver = tf.train.Saver()
        
        ## init network var ##
        self.sess.run(self.init_op)


    def weight_variable(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        fan_out = np.prod(shape[1:])
        std = math.sqrt(3.0 / (fan_in + fan_out))
        initial = tf.truncated_normal(shape, stddev = std)
        #initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        fan_out = np.prod(shape[1:])
        std = math.sqrt(3.0 / (fan_in + fan_out))
        #initial = tf.random_uniform([shape[-1]], minval=(-std), maxval=std)
        initial = tf.constant(0.01, shape=[shape[-1]])
        return tf.Variable(initial, name=name)

    def conv2d_relu(self, x, kernel, stride, name):
        with tf.name_scope(name):
            ## training ##
            W = self.weight_variable(kernel, name='w')
            b = self.bias_variable(kernel, name='b')
            activation = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME', name='conv2d') + b, name='relu')

            tf.histogram_summary(name+'w', W)
            tf.histogram_summary(name+'b', b)

        return activation

    def FC_layer(self, x, kernel, name, activate):
        with tf.name_scope(name):
            ## training ##
            W = self.weight_variable(kernel, name='w')
            b = self.bias_variable(kernel, name='b')
            z_out = tf.matmul(x, W) + b

            tf.histogram_summary(name+'w', W)
            tf.histogram_summary(name+'b', b)

            if activate:
                activation = tf.nn.relu(z_out, name='relu')
            else:
                activation = z_out
        
        return activation

    def max_pool_2x2(self, x, name):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", name=name+'pool')


    def createRNN(self):
        with self.sess.graph.as_default():
            self.prob = tf.placeholder("float", name="keep_prob")
            # input layer #
            with tf.name_scope("input"):
                self.s = tf.placeholder("float", [None, DAYS_RANGE, INPUT_DIM], name='input_state')
                s_tran = tf.transpose(self.s, [1, 0, 2])
                s_re = tf.reshape(s_tran, [-1, INPUT_DIM])
                s_list = tf.split(0, DAYS_RANGE, s_re) ## split s to DAYS_RANGE tensor of shape [BATCH, INPUT_DIM]

            lstm_cell = rnn_cell.LSTMCell(1024, use_peepholes=True, forget_bias=1.0, state_is_tuple=True)
            lstm_drop = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.prob)
            lstm_stack = rnn_cell.MultiRNNCell([lstm_cell]*3, state_is_tuple=True)

            lstm_output, hidden_states = rnn.rnn(lstm_stack, s_list, dtype='float', scope='LSTMStack') # out: [timestep, batch, hidden], state: [cell, c+h, batch, hidden]
                
            h_fc1 = self.FC_layer(lstm_output[-1], [1024, 1024], name='h_fc1', activate=True)
            h_fc1_d = tf.nn.dropout(h_fc1, keep_prob=self.prob, name='h_fc1_drop')
            h_fc2 = self.FC_layer(h_fc1_d, [1024, ACTIONS], name='h_fc2', activate=False)

            # output layer #
            self.pred_action = tf.nn.softmax(h_fc2)


    def createCNN(self):
       
        with self.sess.graph.as_default():
            # input layer #
            with tf.name_scope('input'):
                self.s = tf.placeholder("float", [None, DAYS_RANGE, INPUT_DIM], name='input_state')
                s_4d = tf.reshape(self.s, [-1, DAYS_RANGE, INPUT_DIM, 1])
            
            h_conv1_fx = self.conv2d_relu(s_4d, [3, INPUT_DIM, 1, 32], 1, name='h_conv1_fx')
            '''h_conv1_2x = self.conv2d_relu(s_4d, [5, 2, 1, 32], 1, name='h_conv1_2x')
            h_conv1_1x = self.conv2d_relu(s_4d, [5, 1, 1, 32], 1, name='h_conv1_1x')
            
            # depth reduction of inception block #
            h_conv1_fx_inc = self.conv2d_relu(h_conv1_fx, [1, 1, 32, 12], 1, name='h_conv1_fx_inc')
            h_conv1_2x_inc = self.conv2d_relu(h_conv1_2x, [1, 1, 32, 10], 1, name='h_conv1_2x_inc')
            h_conv1_1x_inc = self.conv2d_relu(h_conv1_1x, [1, 1, 32, 10], 1, name='h_conv1_1x_inc')
            
            h_conv1_dcon = tf.concat(3, [h_conv1_1x_inc, h_conv1_2x_inc, h_conv1_fx_inc])'''
            
            h_conv2 = self.conv2d_relu(h_conv1_fx, [3, INPUT_DIM, 32, 64], 1, name='h_conv2')
            h_conv3 = self.conv2d_relu(h_conv2, [2, INPUT_DIM, 64, 64], 1, name='h_conv3')
            
            conv3_flat = tf.reshape(h_conv3, [-1, INPUT_DIM*DAYS_RANGE*64], name='flatten')
            
            h_fc1 = self.FC_layer(conv3_flat, [INPUT_DIM*DAYS_RANGE*64, 1024], name='h_fc1', activate=True)
            h_fc2 = self.FC_layer(h_fc1, [1024, ACTIONS], name='h_fc2', activate=False)

            # output layer #
            self.pred_action = tf.nn.softmax(h_fc2)


    def createOP(self):

        with self.sess.graph.as_default():
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.rev = tf.placeholder("float", [None], 'rev')
            with tf.name_scope('train'):
                self.corr_action = tf.placeholder("float", [None, ACTIONS], 'corr_action')
                
                l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) ## l2 loss
                print [v.name for v in tf.trainable_variables()]

                cross_entropy = -tf.reduce_mean(tf.reduce_sum(tf.log(self.pred_action) * self.corr_action, reduction_indices=1))
                self.cost = cross_entropy + 0.0000001*l2_loss ## total cost ##
                
                # training op #
                start_l_rate = 0.0001
                decay_step = 100000
                decay_rate = 0.5
                learning_rate = tf.train.exponential_decay(start_l_rate, self.global_step, decay_step, decay_rate, staircase=False)
                grad_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
                self.train_step = tf.contrib.layers.optimize_loss(loss=self.cost, 
                                                                  global_step=self.global_step, 
                                                                  learning_rate=0.0001,
                                                                  optimizer=grad_op, 
                                                                  clip_gradients=1)
                tf.scalar_summary('learning_rate', learning_rate)
                tf.scalar_summary('l2_loss', l2_loss)

            with tf.name_scope('accuracy'):
                self.corr_or_not = tf.cast(tf.equal(tf.argmax(self.corr_action, 1), tf.argmax(self.pred_action, 1)), tf.float32)
                self.accuracy = tf.reduce_mean(self.corr_or_not)
                self.batch_rev = tf.reduce_sum(tf.mul(self.corr_or_not*2-1, self.rev))
        
        ## training op ##
        with self.sess.graph.as_default():
            self.merged = tf.merge_all_summaries()
            self.init_op = tf.initialize_all_variables()

        ## test acc op ##
        with self.g_record.as_default():
            with tf.name_scope('test'):
                self.testAcc = tf.placeholder('float', name='accuracy')
                self.testRev = tf.placeholder('float', name='test_revenue')
                self.trainRev = tf.placeholder('float', name='train_revenue')

                testAcc_summ = tf.scalar_summary('accuracy', self.testAcc)
                testRev_summ = tf.scalar_summary('test_revenue', self.testRev)
                trainRev_summ = tf.scalar_summary('train_revenue', self.trainRev)
            self.merged_test = tf.merge_summary([testAcc_summ, testRev_summ, trainRev_summ])
        

    def TestAcc(self, step):
        
        if step % 10000 == 0 or self.test_Market == None:
            self.test_Market, self.test_comp_name = self.MarketMgr.NextTestMkt()
        
        ## calc ACC ##
        accuracy = 0.
        tot_reward = 0.
        batch_num = 0.
        while True:
            test_data, test_label, test_r_t, terminal = self.test_Market.TestRNNBatch(1024)
            acc, batch_rev = self.sess.run([self.accuracy, self.batch_rev],
                                           feed_dict={self.s : test_data, self.corr_action : test_label, self.rev : test_r_t, self.prob : 1.})
            accuracy += acc
            tot_reward += batch_rev
            batch_num += 1.
            if terminal:
                break
        accuracy = accuracy / batch_num
        return accuracy, tot_reward


    def Inference(self):
        ## loading networks ##
        checkpoint = tf.train.get_checkpoint_state(SAVED_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
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
                print "Invalid idx "+str(idx)
        print 'Testing '+comp_name
        
        plt.ion()
        color_map = ['g-', 'r-']
        ## calc ACC ##
        accuracy = 0.
        tot_reward = 0.
        batch_num = 0.
        Close_p = self.test_Market.GetClose()[DAYS_RANGE-1:]
        full_corr_or_not = []
        while True:
            test_data, test_label, test_r_t, terminal = self.test_Market.TestRNNBatch(1024)
            acc, batch_rev, corr_or_not = self.sess.run([self.accuracy, self.batch_rev, self.corr_or_not],
                                                        feed_dict={self.s : test_data, self.corr_action : test_label, self.rev : test_r_t, self.prob : 1.})
            accuracy += acc
            tot_reward += batch_rev
            batch_num += 1.
            assert len(corr_or_not.shape) == 1
            full_corr_or_not = full_corr_or_not + list(corr_or_not)
            if terminal:
                break
        accuracy = accuracy / batch_num
        print 'Testing '+comp_name+' End' \
              '  ACC: %.4f' % accuracy, \
              '  REV: %.4f' % tot_reward
        self.sess_record.close()
        self.sess.close()
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
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

        ## start training ##
        outfile = sys.stdout

        best_acc = -1
        outfile.write('\n' + "-----------------TRAINING-----------------" + '\n')
        total_start = time.time()
        ep_start = time.time()
        step = 0
        epoch = 0
        try:
            while True:
                step_start = time.time()
                tot_reward = 0.
                self.Market, _ = self.MarketMgr.NextTrainMkt()
                while True:
                    batch_x, batch_y, batch_r_t, terminal = self.Market.NextRNNBatch(BATCH)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    # perform gradient step #
                    _, summary, loss, batch_rev = self.sess.run([self.train_step, self.merged, self.cost, self.batch_rev], 
                                                                feed_dict = {self.s : batch_x, self.corr_action : batch_y, self.rev : batch_r_t, self.prob : 0.85}, 
                                                                options=run_options, 
                                                                run_metadata=run_metadata)
                    if (step+1) % 10 == 0:
                        train_writer.add_run_metadata(run_metadata, str(step+1))
                        train_writer.add_summary(summary, step+1)
                    tot_reward += batch_rev
                    outfile.write("    step: " + str(step+1).ljust(len(str(10**6))+4))
                    outfile.write("loss: " + "{:.4f}".format(loss).ljust(6+4) )
                    outfile.write("Total Reward: " + "{:.2f}".format(tot_reward).ljust(6+4) )
                    outfile.write("elapsed time: " + "{:.2f}".format(time.time() - step_start).rjust(5) + ' secs/batch \n')
                    step_start = time.time()
                    step += 1
                    if terminal:
                        break
                outfile.write("Epoch: " + str(epoch+1).ljust(len(str(100))+11))
                outfile.write("Finished        ")
                outfile.write("Total Reward: " + "{:.2f}".format(tot_reward).ljust(6+4) )
                outfile.write("elapsed time: " + "{:.2f}".format(time.time() - ep_start).rjust(5) + ' secs/epoch \n')
                ## testing ##
                test_start = time.time()
                accuracy, test_reward = self.TestAcc(step)
                record_summary = self.sess_record.run(self.merged_test,
                                                      feed_dict={self.testAcc : accuracy, self.testRev : test_reward, self.trainRev : tot_reward})
                record_writer.add_summary(record_summary, step+1)
                outfile.write("TESTING:             ")
                outfile.write("ACC : " + "{:.4f}".format(accuracy).ljust(6+4) )
                outfile.write("Total Reward: " + "{:.2f}".format(test_reward).ljust(6+4) )
                outfile.write("elapsed time: " + "{:.2f}".format(time.time() - test_start).rjust(5) + ' secs/test \n\n')
                
                if accuracy > best_acc:
                    best_acc = accuracy
                self.saver.save(self.sess, SAVED_PATH+'/' + "TSMC" + '-lstm', global_step=step)
                epoch += 1
                ep_start = time.time()
        except KeyboardInterrupt:
            outfile.write("\b\b\bStop Training...\n")
            test_start = time.time()
            accuracy, test_reward = self.TestAcc(step)
            outfile.write("\nFINAL:   ")
            outfile.write("ACC : " + "{:.4f}".format(accuracy).ljust(6+4) )
            outfile.write("Total Reward: " + "{:.2f}".format(test_reward).ljust(6+4) )
            outfile.write("elapsed time: " + "{:.2f}".format(time.time() - test_start).rjust(5) + ' secs/test \n')
            if accuracy > best_acc:
                best_acc = accuracy
                self.saver.save(self.sess, SAVED_PATH+'/' + "TSMC" + '-lstm', global_step=step)
            print "Total Time:", time.time() - total_start



def main(mode):
    StockModel = MODEL()
    if mode == "train":
        StockModel.trainNetwork()
    elif mode == "test":
        StockModel.Inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Supervised Learning")
    parser.add_argument("-m","--mode", help="mode of DQN, default to train", choices=("train", "test"), default="train")
    args = parser.parse_args()
    main(args.mode)
