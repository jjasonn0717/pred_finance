## pred_finance
> stock price prediction using machine learning

#### stock_DQN.py
 - Deep Q-network implementation
 - DQN model type include CNN and LSTM
 - Target-freezing and DDQN are also implemented
 - ```python stock_DQN.py``` for training, which will automatically search for pre-trained parameters to load
 - ```python stock_DQN.py -m test``` for inference, which will show total reward and accuracy
 - While infering, an action to close price figure will also be generated. Color red stand for correct action, color green stand for wrong action
 - several training arguments can be modified inside the python script

#### stock_DQN_prioritized.py
 - implement DQN with prioritized experiency replay. [Reference](http://arxiv.org/pdf/1511.05952v4.pdf)
 - Only CNN network are used in the current script
 - Usage and features are same as stock_DQN.py

#### stock_supervised.py
 - Supervised Deep Learning implementation
 - CNN and LSTM are included
 - Usage and features are same as stock_DQN.py

#### save_stock.py
 - Saving stock data from yahoo_finance
 - Calculating several index
 - Currently, Index includes: High Low Open Close d_Close RSI9 RSI15 MA5 MA20 MA60 d_CO d_HL Adj_Close Volume VA/D d_VA/D %R8 %R21 DIF DEM d_MA5_20
 - Before running th script,  create a folder stock_data, where csv file will be stored
 - Usage: ```python save_stock.py NAME INDEX [-h] [-e END_YEAR] [-s START_YEAR] [-r RATIO]```
 - NAME: the name of csv file to store
 - INDEX: stock index of the company
 - --start: the data will start from START_YEAR
 - --end: the data will end with END_YEAR
 - --ratio: when apply the ratio argument, data will be split into train and test csv file, which NAME_train.csv contains RATIO percent of data
 - see ```python save_stock.py for more info```

#### draw_stock.py
 - draw the stock data from applied csv file
 - Usage: ```python draw_stock.py FILE [-h]```
 - see ```python draw_stock.py -h``` for more info

#### stock_state.py
 - implement class Stock_state
 - can be used to generate next state for given action
 - can be used to generate random-shuffle data batch for supervised learning
 - can be used to generate none-random-shuffle test batch for supervised learning
 - can also perform similar function as draw_stock.py
 - usage: ```python stock_state.py COMP_NAME```, e.g ```python stock_state.py TSMC```
