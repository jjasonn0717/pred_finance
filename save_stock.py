import sys
import time
from yahoo_finance import Share
import numpy as np
import datetime
import argparse

if __name__ == "__main__":
    
    ## argument parse ##
    parser = argparse.ArgumentParser(description="Calc analysis and Saving data from Yahoo Finance!")	
    parser.add_argument("name", help="company name of applied index")
    parser.add_argument("index", help="Stock index of the company")
    parser.add_argument("--end",'-e', help="ending date, yyyy-mm-dd/today default to today", default='today')
    parser.add_argument("--start", '-s', help="starting date, yyyy-mm-dd, default to 1990-1-1", default='1990-1-1')
    parser.add_argument("--ratio", '-r', help="ratio of training data, default to 1.0, which stand for no splittng and only one data file will be generated",
                        type=float,
                        default=1.0)
    args = parser.parse_args()

    name = args.name
    Stock = Share(args.index)
    if args.end == "today":
        end = str(datetime.date.today())
    else:
        end = args.end
    start = args.start

    if args.ratio < 0. or args.ratio > 1.:
        raise "Invalid ratio argumant! Should be in range 0.~1."
    else:
        TRAIN_RATIO = args.ratio

    history_data = Stock.get_historical(start, end)
    history_data.reverse()
    ## delete data which volume equals zero ##
    history_data = [data for data in history_data if not float(data['Volume']) == 0]
    data_size = len(history_data)
    daily_d_close = np.array([], dtype='float')
    daily_RSI9 = np.array([], dtype='float')
    daily_RSI15 = np.array([], dtype='float')
    daily_VA_D = np.array([], dtype='float')

    if TRAIN_RATIO == 1.0:
        train_file = open('./stock_data/'+name+'.csv', 'w')
    else:
        train_file = open('./stock_data/'+name+'_train.csv', 'w')
        test_file = open('./stock_data/'+name+'_test.csv', 'w')


    print "Days in total:", data_size
    print "slice data into..."
    print "Train:", int(data_size*TRAIN_RATIO), "  Test:", data_size - int(data_size*TRAIN_RATIO)

    train_file.write(Stock.get_info()['symbol'] + ',' + 'High,Low,Open,Close,d_Close,RSI9,RSI15,MA5,MA20,MA60,d_CO,d_HL,Adj_Close,Volume,VA/D,d_VA/D,%R8,%R21,DIF,DEM,d_MA5/20\n')
    if not TRAIN_RATIO == 1.0:
        test_file.write(Stock.get_info()['symbol'] + ',' + 'High,Low,Open,Close,d_Close,RSI9,RSI15,MA5,MA20,MA60,d_CO,d_HL,Adj_Close,Volume,VA/D,d_VA/D,%R8,%R21,DIF,DEM,d_MA5/20\n')

    VA_D_tm1 = 0.
    EMA12_tm1 = 0.
    EMA26_tm1 = 0.
    DEM_tm1 = 0.
    for i in xrange(data_size):
        data = history_data[i]
        volume = float(data['Volume'])
        d_CO = float(data['Close']) - float(data['Open'])
        d_HL = float(data['High']) - float(data['Low'])
        d_CL = float(data['Close']) - float(data['Low'])
        d_HC = float(data['High']) - float(data['Close'])

        ## calc d_close ##
        if i == 0:
            d_close = 0.
        else:
            d_close = float(data['Close']) - float(history_data[i-1]['Close'])
        daily_d_close = np.append(daily_d_close, [d_close])

        ## calc RSI9 ##
        if i < 9:
            sig_abs_d_close = np.sum(np.absolute(daily_d_close))
            sig_pos_d_close = np.sum(np.multiply(daily_d_close, (daily_d_close > 0)))
        else:
            sig_abs_d_close = np.sum(np.absolute(daily_d_close[i-9+1:]))
            sig_pos_d_close = np.sum(np.multiply(daily_d_close[i-9+1:], (daily_d_close[i-9+1:] > 0)))
        assert sig_abs_d_close >= 0 and sig_pos_d_close >= 0
        if sig_abs_d_close == 0:
            daily_RSI9 = np.append(daily_RSI9, [0.])
        else:
            daily_RSI9 = np.append(daily_RSI9, [sig_pos_d_close / sig_abs_d_close])

        ## calc RSI15 ##
        if i < 15:
            sig_abs_d_close = np.sum(np.absolute(daily_d_close))
            sig_pos_d_close = np.sum(np.multiply(daily_d_close, (daily_d_close > 0)))
        else:
            sig_abs_d_close = np.sum(np.absolute(daily_d_close[i-15+1:]))
            sig_pos_d_close = np.sum(np.multiply(daily_d_close[i-15+1:], (daily_d_close[i-15+1:] > 0)))
        assert sig_abs_d_close >= 0 and sig_pos_d_close >= 0
        if sig_abs_d_close == 0:
            daily_RSI15 = np.append(daily_RSI15, [0.])
        else:
            daily_RSI15 = np.append(daily_RSI15, [sig_pos_d_close / sig_abs_d_close])

        ## calc MA5 ##
        MA5 = 0.
        count = 0.
        for t in xrange(5):
            if (i - t) >= 0:
                MA5 += float(history_data[i - t]['Close'])
                count += 1.
        MA5 /= count
        
        ## calc MA20 ##
        MA20 = 0.
        count = 0.
        for t in xrange(20):
            if (i - t) >= 0:
                MA20 += float(history_data[i - t]['Close'])
                count += 1.
        MA20 /= count

        ## calc MA60 ##
        MA60 = 0.
        count = 0.
        for t in xrange(60):
            if (i - t) >= 0:
                MA60 += float(history_data[i - t]['Close'])
                count += 1.
        MA60 /= count

        ## calc VA/D ##
        if i == 0:
            VA_D = volume
        else:
            if not d_HL == 0:
                VA_D = daily_VA_D[-1] + ((d_CL - d_HC) / d_HL) * volume
            else:
                VA_D = daily_VA_D[-1]
        daily_VA_D = np.append(daily_VA_D, [VA_D])
        if i == 0:
            d_VA_D = 0.
            VA_D_tm1 = VA_D
        else:
            d_VA_D = (VA_D-VA_D_tm1)/1000000
            VA_D_tm1 = VA_D

        ## calc piR8 ##
        if i < 8:
            min_dclose = np.min(daily_d_close[:])
            max_dclose = np.max(daily_d_close[:])
        else:
            min_dclose = np.min(daily_d_close[i-8+1:])
            max_dclose = np.max(daily_d_close[i-8+1:])
        piR8 = (d_close - max_dclose)/(max_dclose-min_dclose) if not (max_dclose - min_dclose) == 0. else 0.

        ## calc piR21 ##
        if i < 21:
            min_dclose = np.min(daily_d_close[:])
            max_dclose = np.max(daily_d_close[:])
        else:
            min_dclose = np.min(daily_d_close[i-21+1:])
            max_dclose = np.max(daily_d_close[i-21+1:])
        piR21 = (d_close - max_dclose)/(max_dclose-min_dclose) if not (max_dclose - min_dclose) == 0. else 0.
        
        ## calc MACD ##
        DI = (float(data['High']) + float(data['Low']) + 2*float(data['Close'])) / 4.
        if  i == 0:
            EMA12 = DI
            EMA26 = DI
        else:
            EMA12 = (11./13.)*EMA12_tm1 + (2./13.)*DI
            EMA26 = (25./27.)*EMA26_tm1 + (2./27.)*DI
        EMA12_tm1 = EMA12
        EMA26_tm1 = EMA26
        DIF = EMA12 - EMA26
        if i == 0:
            DEM = DIF
        else:
            DEM = (8./10.)*DEM_tm1 + (2./10.)*DIF
        DEM_tm1 = DEM


        if volume == 0:
            raise "volume == 0", data['Date']
        if i < int(data_size*TRAIN_RATIO):
            train_file.write(data['Date'] + ',' +
                             data['High'] + ',' +
                             data['Low'] + ',' +
                             data['Open'] + ',' +
                             data['Close'] + ',' +
                             str(d_close) + ',' +
                             str(daily_RSI9[i]) + ',' +
                             str(daily_RSI15[i]) + ',' +
                             #str(daily_RSI9[i]-daily_RSI15[i]) + ',' +
                             str(MA5) + ',' +
                             str(MA20) + ',' +
                             str(MA60) + ',' +
                             str(d_CO) + ',' +
                             str(d_HL) + ',' +
                             data['Adj_Close'] + ',' +
                             str(volume/1000000) + ',' +
                             str(VA_D/1000000) + ',' +
                             str(d_VA_D) + ',' +
                             str(piR8) + ',' +
                             str(piR21) + ',' +
                             str(DIF) + ',' +
                             str(DEM) + ',' +
                             str(MA5-MA20) + '\n')
        else:
            test_file.write(data['Date'] + ',' +
                             data['High'] + ',' +
                             data['Low'] + ',' +
                             data['Open'] + ',' +
                             data['Close'] + ',' +
                             str(d_close) + ',' +
                             str(daily_RSI9[i]) + ',' +
                             str(daily_RSI15[i]) + ',' +
                             #str(daily_RSI9[i]-daily_RSI15[i]) + ',' +
                             str(MA5) + ',' +
                             str(MA20) + ',' +
                             str(MA60) + ',' +
                             str(d_CO) + ',' +
                             str(d_HL) + ',' +
                             data['Adj_Close'] + ',' +
                             str(volume/1000000) + ',' +
                             str(VA_D/1000000) + ',' +
                             str(d_VA_D) + ',' +
                             str(piR8) + ',' +
                             str(piR21) + ',' +
                             str(DIF) + ',' +
                             str(DEM) + ',' +
                             str(MA5-MA20) + '\n')

    train_file.close()
    if not TRAIN_RATIO == 1.0:
        test_file.close()
