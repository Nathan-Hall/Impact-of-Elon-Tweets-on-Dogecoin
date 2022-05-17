import requests
import pandas as pd
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from scipy import stats
from plotly.offline import plot
import statsmodels.api as sm


def minutes_unix(minutes):
    unix = minutes*60*1000
    
    return unix


def get_start_end(window, tweet_timestamp):
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


def calc_abnormal_ret(df, tweet_df, window, event_count, specs, recog, tweet_filename):
    #Preparation
    df = df.reset_index()
    total_time = window*2+1
    event_start = specs['event_start']
    event_end = specs['event_end']
    grouped_ar, mean_ar, cum_ar, hr_log_ret, hr_vol = [],[],[],[],[]
    pd.options.mode.chained_assignment = None 
    
    #Create each column so it is possible to assign using [x:y]
    df['Return']=0
    df['mean']=0 
    df['sd']=0     
    df['CAR']=0
    df['cumsum']=0
    
    #prep car
    #car_freq = specs['car_freq']
    #car_list = [1,2,5,10,15]
    # car_list.extend([j*car_freq for j in range(1,6)])
    car_list = [j for j in range(1,60)]
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
        df['cumsum'].iloc[window_start:window_end] = df[window_start:window_end].Return.cumsum()
        
        #1 hour log return and volume:
        hr_log_ret.append(df['Return'].iloc[tweet_start-61:tweet_start-1].cumsum().iloc[-1])
        hr_vol.append(df['Volume'].iloc[tweet_start-61:tweet_start-1].cumsum().iloc[-1])
        
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
    
    #append 1-hour log returns (momentum)
    tweet_df['hour_ret'] = hr_log_ret
    tweet_df['hour_vol'] = hr_vol
    
    #Add recognisability
    if tweet_filename == "Elon Crypto Tweets.csv":
        tweet_df['Recognisability'] = recog
    
    return df, df_ar, caar_df, wilcoxon_res, tweet_df


def get_mcap(tweet_df, specs_cg):
    symbol_cg = specs_cg['symbol']
    currency = specs_cg['currency'] 
    mcap = []
    print('Getting Market Cap')
    for i in tweet_df.Timestamp:
        first = i
        second = i + 86400
        url='https://api.coingecko.com/api/v3/coins/%s/market_chart/range?vs_currency=%s&from=%s&to=%s&interval=daily' % (symbol_cg, currency, first, second)
        mcap.append(requests.get(url).json()['market_caps'][0][1])
        
    tweet_df['Market_Cap'] = mcap
    print('Market cap data retrieved')
    
    return tweet_df


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


def regression(tweet_df, months):
    tweet_df = tweet_df.rename_axis('Tweet_Count').reset_index()
    tweet_df.replace(('yes', 'no'), (1, 0), inplace=True)
    tweet_df['Months'] = months
    tweet_df['Interaction'] = tweet_df['Market_Cap']*tweet_df['Tweet_Count']
    
    CAR_list = tweet_df.filter(like='CAR').columns

    params, pvalues = [],[]
    for val in CAR_list:
        X = tweet_df[['Market_Cap', 'Recognisability', 'hour_vol', 'hour_ret', 
                      'Link','Picture','Video','Tweet_Count', 'Interaction', 'Months']]
        y = tweet_df[val]
        model = sm.OLS(y, X).fit()
        
        params.append(list(model.params))
        pvalues.append(list(model.pvalues))
        
    names = list(dict(model.params))
    names_p = [name + '_p' for name in names]

    plot_df = pd.DataFrame(params)
    plot_df.columns = names
    plot_df[names_p] = pvalues

    for coef_name, p_name in zip(names,names_p):
        plot_coef(plot_df, coef_name, p_name)


def autocor(df, event_count, specs, window):
    event_start = specs['event_start']
    total_time = window*2+1
    df['Lag'] = df['Return'].shift(1).fillna(method='bfill')
    df['dummy'] = 0
    df2 = df
    
    dummy_coef, int_coef, lag_coef, dummy_p, int_p, lag_p = [],[],[],[],[],[]
    
    for i in range(event_count):  
        window_start = i*total_time
        tweet_start = window_start + window
        window_end = tweet_start + window
        df['dummy'].iloc[tweet_start:window_end] = 1
        df['Interaction'] = df['Lag'] * df['dummy']
        df2 = df.iloc[tweet_start-180:tweet_start+20]
        
        X = df2[['Lag', 'Interaction', 'dummy']]
        X = sm.add_constant(X)
        y = df2['Return']
        model = sm.OLS(y, X).fit()
        model.summary()
        
        dummy_coef.append(model.params.dummy)
        int_coef.append(model.params.Interaction)
        lag_coef.append(model.params.Lag)
        
        dummy_p.append(model.pvalues.loc['dummy'])
        int_p.append(model.pvalues.loc['Interaction'])
        lag_p.append(model.pvalues.loc['Lag'])

    plot_df2 = pd.DataFrame([dummy_coef, int_coef, lag_coef, dummy_p, int_p, lag_p]).T.reset_index(drop=True)
    plot_df2.columns = ['dummy_coef', 'int_coef', 'lag_coef', 'dummy_p', 'int_p', 'lag_p']
    plot_df2[['dummy_coef','int_coef', 'lag_coef']].plot()
    
    
def plot_coef(plot_df, coef_name, p_name):       
    fig,ax = plt.subplots()
    ax.plot(plot_df.index, plot_df[coef_name], color='Blue', label= (f"{coef_name} Coefficient"))
    fig.suptitle(f"Changing Regression Coefficient of {coef_name} on CAR")
    ax.set_xlabel('Minutes After Tweet')
    ax.set_ylabel(coef_name)
    ax2 = ax.twinx()
    ax2.plot(plot_df.index, plot_df[p_name], color='Red', label=(f"{coef_name} P-value"))
    ax2.set_ylabel(f"{coef_name} P-value")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    
    fig.savefig(f"Graphs/{coef_name}.png")

def main():
    configs = json.load(open('config.json', 'r'))
    tweet_filename = configs['data']['filename']
    tweet_df = get_tweets(tweet_filename)
    
    specs = configs['specs'] 
    recog = configs['extra']['recognisability']
    months = configs['extra']['months']
    specs_bin = configs['specs_bin']
    specs_ftx = configs['specs_ftx']
    window = specs['window']
    
    data_df, event_count = get_data(tweet_df, specs, specs_bin, specs_ftx, window)
    
    df, df_ar, caar_df, wilcoxon_res, tweet_df = calc_abnormal_ret(data_df, tweet_df, window, event_count, specs, recog, tweet_filename)
    
    #get market cap for each day from coingecko api:
    specs_cg = configs['specs_coingecko']
    tweet_df = get_mcap(tweet_df, specs_cg)
    
    #regressions
    regression(tweet_df, months)
    
    #to excel
    filename_out = configs['output']['filename']
    excel_create(filename_out, df, df_ar, caar_df, wilcoxon_res, tweet_df)
    
    
if __name__ == "__main__":
    main()
