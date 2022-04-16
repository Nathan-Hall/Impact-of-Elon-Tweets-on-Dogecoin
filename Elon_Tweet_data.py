import requests
import pandas as pd
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from scipy import stats
from plotly.offline import plot


def minutes_unix(minutes):
    unix = minutes*60*1000
    
    return unix


def get_start_end(window, tweet_timestamp):
    window = 480
    start_end = []
    for i, val in enumerate(tweet_timestamp):
        window_unix = minutes_unix(window)
        start = val - window_unix
        end = val + window_unix
        start_end.append([start, end])
    
    return start_end


def get_tweets(filename):
    
    #Get Elon Tweets from csv
    tweet_df = pd.read_csv(filename)
    
    #Convert dates to dt formaate/timestamp and convert from berlin timezone to utc (exchange timezone)
    tweet_df['Datetime'] = pd.to_datetime(tweet_df.Date.astype(str) + ' ' + tweet_df.Time.astype(str), format='%d/%m/%Y %I:%M:%S %p')
    tweet_df['Datetime'] = tweet_df['Datetime'].dt.tz_localize('Europe/Berlin').dt.tz_convert('UTC')
    tweet_df['Timestamp'] = tweet_df.Datetime.values.astype(np.int64) // 10 ** 9
    
    #organise df
    tweet_df = tweet_df[(tweet_df['Datetime'].dt.year > 2019)]
    tweet_df = tweet_df[(tweet_df['Coin'].str.contains('DOGE') == True)]
    tweet_df = tweet_df.dropna(subset=['Event'])
    tweet_df = tweet_df.drop(labels=['No', 'Event', 'Date', 'Time'], axis=1)
    tweet_df = tweet_df.reset_index(drop=True)
    
    return tweet_df
    


def get_data(tweet_df, specs, specs_bin, specs_ftx, window):
    #Get data from binance, if binance data is missing due to server maintenance then get data from FTX api
    print('Getting data')
    symbol_bin = specs_bin['symbol']
    interval_bin = specs_bin['interval']
    
    symbol_ftx = specs_ftx['symbol']
    interval_ftx = specs_ftx['interval']
    
    tweet_timestamp = (tweet_df['Timestamp']*1000).tolist()
    start_end = get_start_end(window, tweet_timestamp)
    event_count = 0
    data = []
    data2 = []
    for i in range(len(tweet_timestamp)):
        first = start_end[i][0]
        second = start_end[i][1]
        url='https://api.binance.com/api/v3/klines?symbol=%s&interval=%s&startTime=%s&endTime=%s&limit=%d' % (symbol_bin, interval_bin, first, second, 1000)
        data_temp = requests.get(url).json()
        
        #Make sure no missing observations
        if len(data_temp) == window*2 + 1:
            data.extend(data_temp)
            event_count +=1
        else:
            first = first // 1000
            second = second // 1000
            url2='https://ftx.com/api/markets/%s/candles?resolution=%s&start_time=%s&end_time=%s' % (symbol_ftx, interval_ftx, first, second)
            data_temp2 = requests.get(url2).json()['result']
            if len(data_temp2) == window*2 + 1:
                data2.extend(data_temp2)
                event_count +=1     
            
    if len(data) > 1:
        df = pd.DataFrame(data)
        df = df.iloc[:, [0,4,5]]
        df.columns = ['Time', 'Close','Volume']
        df = df.set_index('Time')
        df.index = pd.to_datetime(df.index, unit='ms')
        df = df.astype(float)
        
        if len(data2) > 1:
            df2 = pd.DataFrame(data2)
            df2 = df2.loc[:, ['time', 'close']]
            df2 = df2.rename(columns={'close':'Close'})
            df2['Volume'] = np.nan
            df2 = df2.set_index('time')
            df2.index = pd.to_datetime(df2.index, unit='ms')
            df2 = df2.astype(float)         
            df = pd.concat([df, df2], axis=0)
            df = df.sort_index(ascending=True)
    
    print('Data retrieved')
    return df, event_count


def calc_abnormal_ret(df, tweet_df, window, event_count, specs):
    #Preparation
    df = df.reset_index()
    total_time = window*2+1
    event_start = specs['event_start']
    event_end = specs['event_end']
    grouped_ar = []
    mean_ar = []
    cum_ar = []
    pd.options.mode.chained_assignment = None 
    
    #Create each column so it is possible to assign using [x:y]
    df['Return']=0
    df['mean']=0 
    df['sd']=0     
    df['CAR']=0
    
    #prep car
    car_freq = specs['car_freq']
    car_list = [j*car_freq for j in range(1,12)]
    df_ar = pd.DataFrame()
    
    for i in range(event_count):
        #Set up parameters of each event
        window_start = i*total_time
        window_end = (i*total_time)+total_time
        exp_ret_window = window_start + window-60
        tweet_start = window_start + window
        
        #Analyse Data
        #Log Returns
        df['Return'].iloc[window_start:window_end] = np.log1p(df[window_start:window_end].Close.pct_change())
        df = df.fillna(method='bfill')
        
        df['mean'].iloc[window_start:window_end] = df[window_start:exp_ret_window].Return.mean()
        df['AR'] = df['Return'] - df['mean']
        df['CAR'].iloc[window_start:window_end] = df[window_start:window_end].AR.cumsum()
        df['sd'].iloc[window_start:window_end] = math.sqrt((np.square(df[window_start:exp_ret_window].AR).sum(axis=0))/(window*2))
        df['AR_tstat'] = df['AR']/df['sd']
        
        #Get first x values for each event for later stat. analysis
        wilcox_amount = 10
        grouped_ar.append([df.Return[tweet_start+j] for j in range(wilcox_amount)])
        mean_ar.append([df['mean'][tweet_start+j] for j in range(wilcox_amount)])
        
        #Get CAR
        cum_ar.append([df.CAR[tweet_start+j] for j in car_list])
        
        #Get AAR, first get AR in event window:
        event_ar = df['AR'][tweet_start+event_start:tweet_start+event_end].tolist()
        df_ar[('Event' + str(i))] = event_ar

    #transpose matrices
    grouped_ar2 = list(map(list, zip(*grouped_ar)))
    mean_ar2 = list(map(list, zip(*mean_ar)))
    cum_ar2 = list(map(list, zip(*cum_ar)))
   
    #Add CAR to tweet_df
    for i, val in enumerate(car_list):
        col_name = 'CAR_' + str(val)
        tweet_df[col_name] = cum_ar2[i]
        
    #AAR
    df_ar['AAR'] = df_ar.iloc[:,0:event_count].mean(axis=1)
    df_ar['index'] = [i for i in range(event_start,event_end)]
    df_ar = df_ar.reset_index()
    
    #CAAR
    caar_index = [30,60,120,180,239]
    caar = [df_ar['AAR'][0:i].cumsum().iloc[-1] for i in caar_index]
    caar_df = pd.DataFrame(caar,caar_index, columns=['CAAR'])
    caar_df.index.name = 'Minutes'
    
    #Wilcoxon_test
    wilcoxon_res = []
    for k in range(len(grouped_ar2)):
        wilcoxon_res.append(stats.wilcoxon(grouped_ar2[k], mean_ar2[k])[1])
    
    #for excel
    df_ar = df_ar.set_index('index')
    df_ar.index.name = 'Minutes'
    
    return df, df_ar, caar_df, wilcoxon_res, tweet_df


def excel_create(filename_out, df, df_ar, caar_df, wilcoxon_res, tweet_df): 
    writer = pd.ExcelWriter(filename_out, engine='xlsxwriter')
    df.to_excel(writer,sheet_name='All Data')
    df_ar.to_excel(writer,sheet_name='AR per event window and AAR')
    
    #Datetime cannot be tz aware in excel
    tweet_df['Datetime'] = tweet_df['Datetime'].dt.tz_localize(None)
    tweet_df.index.name = 'Tweet_Count'
    
    tweet_df.to_excel(writer,sheet_name='Elon Tweets and CAR')
    caar_df.to_excel(writer,sheet_name='CAAR')
    
    wil_df = pd.DataFrame(wilcoxon_res, [i for i in range(1,11)], columns=['P-value'])
    wil_df.index.name = 'Minute'
    wil_df.to_excel(writer, sheet_name='Wilcoxon Test Results')
    
    writer.save()


def main():
    configs = json.load(open('config.json', 'r'))
    tweet_filename = configs['data']['filename']
    tweet_df = get_tweets(tweet_filename)
    
    specs = configs['specs'] 
    specs_bin = configs['specs_bin']
    specs_ftx = configs['specs_ftx']
    window = specs['window']
    
    data_df, event_count = get_data(tweet_df, specs, specs_bin, specs_ftx, window)
    
    df, df_ar, caar_df, wilcoxon_res, tweet_df = calc_abnormal_ret(data_df, tweet_df, window, event_count, specs)
    
    filename_out = configs['output']['filename']
    excel_create(filename_out, df, df_ar, caar_df, wilcoxon_res, tweet_df)
    
    
if __name__ == "__main__":
    main()
