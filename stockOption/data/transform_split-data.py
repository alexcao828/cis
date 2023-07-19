import pandas as pd
import copy 
import numpy as np

root = 'archive/Stocks/'
tickers = ['aapl', 'acn', 'adbe', 'adi', 'adp', 'adsk', 'akam', 'amat',
           'amd', 'anet', 'anss', 'aph', 'avgo', 'br', 'cday', 'cdns', 'cdw',
           'crm', 'csco', 'ctsh', 'dxc', 'enph', 'epam', 'ffiv', 'fis', 'fisv',
           'flt', 'ftnt', 'gen', 'glw', 'gpn', 'hpe', 'hpq', 'ibm', 'intc', 
           'intu', 'it', 'jkhy', 'jnpr', 'keys', 'klac', 'lrcx', 'ma', 'mchp', 
           'mpwr', 'msft', 'msi', 'mu', 'now', 'ntap', 'nvda', 'nxpi', 'on',
           'orcl', 'payc', 'payx', 'ptc', 'pypl', 'qcom', 'qrvo', 'rop', 'sedg', 
           'snps', 'stx', 'swks', 'tdy', 'tel', 'ter', 'trmb', 'txn', 'tyl',
           'v', 'vrsn', 'wdc', 'zbra']
bad = []
for i in range(len(tickers)):
    try:
        stock = pd.read_csv(root+tickers[i]+'.us.txt')
        stock = stock.rename(columns={"Open": tickers[i]+"Open", 
                                      "High": tickers[i]+"High", 
                                      "Low": tickers[i]+"Low", 
                                      "Close": tickers[i]+"Close", 
                                      "Volume": tickers[i]+"Volume"})
        if i == 0:
            open_df = stock[['Date', tickers[i]+"Open"]]
            high_df = stock[['Date', tickers[i]+"High"]]
            low_df = stock[['Date', tickers[i]+"Low"]]
            close_df = stock[['Date', tickers[i]+"Close"]]
            volume_df = stock[['Date', tickers[i]+"Volume"]]
        else:
            open_df_temp = stock[['Date', tickers[i]+"Open"]]
            high_df_temp = stock[['Date', tickers[i]+"High"]]
            low_df_temp = stock[['Date', tickers[i]+"Low"]]
            close_df_temp = stock[['Date', tickers[i]+"Close"]]
            volume_df_temp = stock[['Date', tickers[i]+"Volume"]]
    
            open_df = pd.merge(open_df, open_df_temp, how='outer', on='Date')
            high_df = pd.merge(high_df, high_df_temp, how='outer', on='Date')
            low_df = pd.merge(low_df, low_df_temp, how='outer', on='Date')
            close_df = pd.merge(close_df, close_df_temp, how='outer', on='Date')
            volume_df = pd.merge(volume_df, volume_df_temp, how='outer', on='Date')
    except:
        bad.append(tickers[i])
open_df['Date'] = pd.to_datetime(open_df['Date'])
open_df = open_df.sort_values(by='Date').reset_index(drop=True)

high_df['Date'] = pd.to_datetime(high_df['Date'])
high_df = high_df.sort_values(by='Date').reset_index(drop=True)

low_df['Date'] = pd.to_datetime(low_df['Date'])
low_df = low_df.sort_values(by='Date').reset_index(drop=True)

close_df['Date'] = pd.to_datetime(close_df['Date'])
close_df = close_df.sort_values(by='Date').reset_index(drop=True)

volume_df['Date'] = pd.to_datetime(volume_df['Date'])
volume_df = volume_df.sort_values(by='Date').reset_index(drop=True)

def get_ema(df):
    ema = copy.deepcopy(df)
    for i in range(1, len(ema.columns)):
        ema = ema.rename(columns={ema.columns[i]: ema.columns[i]+"Ema"})
        ema[ema.columns[i]] = ema[ema.columns[i]].ewm(span=8).mean()
    return ema
openEma_df = get_ema(open_df)
highEma_df = get_ema(high_df)
lowEma_df = get_ema(low_df)
closeEma_df = get_ema(close_df)
volumeEma_df = get_ema(volume_df)

def get_bollinger_bands(high, low, close):  
    tpMA = copy.deepcopy(high)
    bollingerLower = copy.deepcopy(high)
    bollingerUpper = copy.deepcopy(high)
    for i in range(1, len(tpMA.columns)):
        name = tpMA.columns[i][0:-4]
        #typical price
        tpMA = tpMA.rename(columns={name+"High": name+"Tpma"})
        bollingerLower = bollingerLower.rename(columns={name+"High": name+"BollingerLower"})
        bollingerUpper = bollingerUpper.rename(columns={name+"High": name+"BollingerUpper"})
        TP = (high[name+'High'] + low[name+'Low'] + close[name+'Close']) / 3
        # takes one column from dataframe
        tpMA[tpMA.columns[i]] = pd.Series((TP.rolling(20, min_periods=1).mean()))
        sigma = TP.rolling(20, min_periods=1).std(ddof=0) 
        bollingerUpper[bollingerUpper.columns[i]] = pd.Series((tpMA[tpMA.columns[i]] + 2 * sigma))
        bollingerLower[bollingerLower.columns[i]] = pd.Series((tpMA[tpMA.columns[i]] - 2 * sigma))
    return tpMA, bollingerUpper, bollingerLower
tpMa_df, bollingerUpper_df, bollingerLower_df  = get_bollinger_bands(high_df, low_df, close_df)

def get_obv(volume, close):
    obv = copy.deepcopy(close)
    for i in range(1, len(obv.columns)):
        name = obv.columns[i][0:-5]
        obv = obv.rename(columns={name+"Close": name+"Obv"})
        obv[obv.columns[i]] = (np.sign(obv[obv.columns[i]].diff()) * volume[name+'Volume']).cumsum()
    return obv
obv_df = get_obv(volume_df, close_df)

def get_ad(close, low, high, volume):
    ad = copy.deepcopy(close)
    for i in range(1, len(ad.columns)):
        name = ad.columns[i][0:-5]
        ad = ad.rename(columns={name+"Close": name+"Ad"})
        mfm = ((close[close.columns[i]] - low[low.columns[i]]) - (high[high.columns[i]] - close[close.columns[i]])) / (high[high.columns[i]] - low[low.columns[i]])
        mfv = mfm * volume[volume.columns[i]]
        ad[ad.columns[i]] = mfv.cumsum()
    return ad
ad_df = get_ad(close_df, low_df, high_df, volume_df)

def get_adx(high, low, close, lookback=14):
    plus_di = copy.deepcopy(high)
    minus_di = copy.deepcopy(high)
    dx = copy.deepcopy(high)
    adx_smooth = copy.deepcopy(high)
    for i in range(1, len(plus_di.columns)):
        name = plus_di.columns[i][0:-4]
        plus_di = plus_di.rename(columns={name+"High": name+"+Di"})
        minus_di = minus_di.rename(columns={name+"High": name+"-Di"})
        dx = dx.rename(columns={name+"High": name+"Dx"})
        adx_smooth = adx_smooth.rename(columns={name+"High": name+"AdxSmooth"})
        plus_dm = high[high.columns[i]].diff()
        minus_dm = -low[low.columns[i]].diff()
        plus_dm[(plus_dm < minus_dm) | (plus_dm < 0)] = 0
        minus_dm[(minus_dm < plus_dm) | (minus_dm < 0)] = 0
        tr1 = pd.DataFrame(high[high.columns[i]] - low[low.columns[i]])
        tr2 = pd.DataFrame(abs(high[high.columns[i]] - close[close.columns[i]].shift(1)))
        tr3 = pd.DataFrame(abs(low[low.columns[i]] - close[close.columns[i]].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.ewm(span = 14).mean()
        plus_di[plus_di.columns[i]] = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
        minus_di[minus_di.columns[i]] = 100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr)
        dx[dx.columns[i]] = (abs(plus_di[plus_di.columns[i]] - minus_di[minus_di.columns[i]]) / abs(plus_di[plus_di.columns[i]] + minus_di[minus_di.columns[i]])) * 100
        adx_smooth[adx_smooth.columns[i]] = dx[dx.columns[i]].ewm(alpha = 1/lookback).mean()
    return plus_di, minus_di, dx, adx_smooth
plusDi_df, minusDi_df, dx_df, adx_df = get_adx(high_df, low_df, close_df)

def get_aroon(high, low, lb=25):
    aroon_up = copy.deepcopy(high)
    aroon_down = copy.deepcopy(high)
    aroon_oscillator = copy.deepcopy(high)
    for i in range(1, len(aroon_up.columns)):
        name = aroon_up.columns[i][0:-4]
        aroon_up = aroon_up.rename(columns={name+"High": name+"AroonUp"})
        aroon_down = aroon_down.rename(columns={name+"High": name+"AroonDown"})
        aroon_oscillator = aroon_oscillator.rename(columns={name+"High": name+"AroonOscillator"})
        aroon_up[aroon_up.columns[i]] = 100 * high[high.columns[i]].rolling(lb, min_periods=1).apply(lambda x: x.argmax()) / lb
        aroon_down[aroon_down.columns[i]] = 100 * low[low.columns[i]].rolling(lb, min_periods=1).apply(lambda x: x.argmin()) / lb
        aroon_oscillator[aroon_oscillator.columns[i]] = aroon_up[aroon_up.columns[i]] - aroon_down[aroon_down.columns[i]]
    return aroon_up, aroon_down, aroon_oscillator
aroonUp_df, aroonDown_df, aroonOscillator_df = get_aroon(high_df, low_df)

def get_macd(close):
    macd = copy.deepcopy(close)
    signal = copy.deepcopy(close)
    for i in range(1, len(macd.columns)):
        name = macd.columns[i][0:-5]
        macd = macd.rename(columns={name+"Close": name+"Macd"})
        signal = signal.rename(columns={name+"Close": name+"MacdSignal"})
        macd[macd.columns[i]] = macd[macd.columns[i]].ewm(span=12).mean() - macd[macd.columns[i]].ewm(span=26).mean()
        signal[signal.columns[i]] = macd[macd.columns[i]].ewm(span=9).mean()
    return macd, signal
macd_df, macdSignal_df = get_macd(close_df)

def get_rsi(close, periods = 14):
    """
    Returns a pd.Series with the relative strength index.
    """
    rsi = copy.deepcopy(close)
    for i in range(1, len(rsi.columns)):
        name = rsi.columns[i][0:-5]
        rsi = rsi.rename(columns={name+"Close": name+"Rsi"})
        close_delta = close[close.columns[i]].diff()
        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
    	# Use exponential moving average
        ma_up = up.ewm(span = periods).mean()
        ma_down = down.ewm(span = periods).mean()
        this_rsi = ma_up / ma_down
        rsi[rsi.columns[i]] = 100 - (100/(1 + this_rsi))
    return rsi
rsi_df = get_rsi(close_df)

def get_stochastic_oscillator(close, high, low):
    fast = copy.deepcopy(close)
    slow = copy.deepcopy(close)
    for i in range(1, len(fast.columns)):
        name = fast.columns[i][0:-5]
        fast = fast.rename(columns={name+"Close": name+"FastStochOsc"})
        slow = slow.rename(columns={name+"Close": name+"SlowStochOsc"})
        high14 = high[high.columns[i]].rolling(14, min_periods=1).max()
        low14 = low[low.columns[i]].rolling(14, min_periods=1).min()
        fast[fast.columns[i]] = (close[close.columns[i]] - low14)*100/(high14 - low14)
        slow[slow.columns[i]] = fast[fast.columns[i]].rolling(3, min_periods=1).mean()
    return fast, slow
fastStochOsc_df, slowStochOsc_df = get_stochastic_oscillator(close_df, high_df, low_df)

def cum_norm(df):
    maxs = np.zeros(df.shape[0])
    mins = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        if i == 0:
            maxs[i] = np.nanmax(df[df.columns[1:]].iloc[i].values)
            mins[i] = np.nanmin(df[df.columns[1:]].iloc[i].values)
        else:
            new_max = np.nanmax(df[df.columns[1:]].iloc[i].values)
            new_min = np.nanmin(df[df.columns[1:]].iloc[i].values)
            
            maxs[i] = np.nanmax(np.asarray([maxs[i-1], new_max]))
            mins[i] = np.nanmin(np.asarray([mins[i-1], new_min]))
    ranges = maxs-mins
    ranges[ranges == 0] = 1
    df[df.columns[1:]] = df[df.columns[1:]].sub(mins, axis='rows').div(ranges, axis='rows')
    return df
Y = copy.deepcopy(close_df)
open_df = cum_norm(open_df)
high_df = cum_norm(high_df)
low_df = cum_norm(low_df)
close_df = cum_norm(close_df)
volume_df = cum_norm(volume_df)
openEma_df = cum_norm(openEma_df)
highEma_df = cum_norm(highEma_df)
lowEma_df = cum_norm(lowEma_df)
closeEma_df = cum_norm(closeEma_df)
volumeEma_df = cum_norm(volumeEma_df)
tpMa_df = cum_norm(tpMa_df)
bollingerUpper_df = cum_norm(bollingerUpper_df)
bollingerLower_df = cum_norm(bollingerLower_df)
obv_df = cum_norm(obv_df)
ad_df = cum_norm(ad_df)
plusDi_df = cum_norm(plusDi_df)
minusDi_df = cum_norm(minusDi_df)
dx_df = cum_norm(dx_df)
adx_df = cum_norm(adx_df)
aroonUp_df = cum_norm(aroonUp_df)
aroonDown_df = cum_norm(aroonDown_df)
aroonOscillator_df = cum_norm(aroonOscillator_df)
macd_df = cum_norm(macd_df)
macdSignal_df = cum_norm(macdSignal_df)
rsi_df = cum_norm(rsi_df)
fastStochOsc_df = cum_norm(fastStochOsc_df)
slowStochOsc_df = cum_norm(slowStochOsc_df)

all_dfs = [Y, open_df, high_df, low_df, close_df, volume_df, openEma_df, highEma_df, 
           lowEma_df, closeEma_df, volumeEma_df, tpMa_df, bollingerUpper_df, 
           bollingerLower_df, obv_df, ad_df, plusDi_df, minusDi_df, dx_df, 
           adx_df, aroonUp_df, aroonDown_df, aroonOscillator_df, macd_df, 
           macdSignal_df, rsi_df, fastStochOsc_df, slowStochOsc_df]
# big_matrix (stocks, T, y/close+features)
T = open_df.shape[0]
num_stocks = len(tickers) - len(bad)
for i in range(len(all_dfs)):
    df = all_dfs[i]
    if i == 0:
        big_matrix = df[df.columns[1:]].values.reshape(T, num_stocks, 1)
    else:
        temp = df[df.columns[1:]].values.reshape(T, num_stocks, 1)
        big_matrix = np.concatenate((big_matrix, temp), axis=2)
big_matrix = np.transpose(big_matrix, (1, 0, 2))

def make_episodes(all_stocks_big_M):
    window = 30
    length_needed = window
    num_stocks = np.shape(all_stocks_big_M)[0]
    num_features = np.shape(all_stocks_big_M)[2]
    train_counter = 0
    val_counter = 0
    test_counter = 0
    train_y = []
    for i in range(num_stocks):
        stock_matrix = all_stocks_big_M[i]
        # remove nans
        stock_matrix = stock_matrix[~np.isnan(stock_matrix).any(axis=1), :]
        num_t_in_this_stock = np.shape(stock_matrix)[0]
        
        #take last year for test, second-to-last for val
        if num_t_in_this_stock >= (365)*3:
            if test_counter == 0:
                test_set = stock_matrix[-(365):].reshape(1, (365), num_features)
                test_counter += 1
            else:
                temp = stock_matrix[-(365):].reshape(1, (365), num_features)
                test_set = np.concatenate((test_set, temp), axis=0)
                test_counter += 1
            stock_matrix = stock_matrix[0:-(365)]
            
            if val_counter == 0:
                val_set = stock_matrix[-(365):].reshape(1, (365), num_features)
                val_counter += 1
            else:
                temp = stock_matrix[-(365):].reshape(1, (365), num_features)
                val_set = np.concatenate((val_set, temp), axis=0)
                val_counter += 1
            stock_matrix = stock_matrix[0:-(365)]
            
        #make train episodes from this stock
        while np.shape(stock_matrix)[0] >= length_needed:
            print(np.shape(stock_matrix)[0], end=' ')
            priceDay1 = stock_matrix[0, 0]
            priceDay30 = stock_matrix[window-1, 0]
            if priceDay30 >= priceDay1:
                train_y.append(1)
            else:
                train_y.append(0)
            
            this_train_sample = np.zeros((1,window, num_features))
            this_train_sample[0, :, :] = stock_matrix[0:window]
            
            if train_counter == 0:
                train_features = this_train_sample
                train_counter += 1
            else:
                train_features = np.concatenate((train_features, this_train_sample), axis=0)
                train_counter += 1 
            stock_matrix = stock_matrix[window:]  
        print('stock ', (i+1), '/', num_stocks, ' done')
    #save
    test_close = test_set[:, :, 0]
    np.save('test_close.npy', test_close)
    test_features = test_set[:, :, 1:]
    np.save('test_features.npy', test_features)
    val_close = val_set[:, :, 0]
    np.save('val_close.npy', val_close)
    val_features = val_set[:, :, 1:]
    np.save('val_features.npy', val_features)
    train_features = train_features[:, :, 1:]
    np.save('train_features.npy', train_features)
    np.save('train_profit.npy', np.asarray(train_y))
make_episodes(big_matrix)
