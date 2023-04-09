from numpy import nan
import talib.abstract as ta
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib

import math
# import talib as ta
# import matplotlib.pyplot as plt

# -------------------------------- UTILS --------------------------------


def minimum(s1, s2) -> float:
    return np.minimum(s1, s2)


def maximum(s1, s2) -> float:
    return np.maximum(s1, s2)


def highestbars(series, n):
    return ta.MAXINDEX(series, n) - n + 1


def lowestbars(series, n):
    return ta.MININDEX(series, n) - n + 1


def nz(series):
    return series.fillna(method='backfill')


def na(series):
    return series is nan


def highest(series, n):
    return series.rolling(n).max()


def lowest(series, n):
    return series.rolling(n).min()


def series(value, df):
    return pd.Series(value, index=df.index)


def change(series):
    return series.diff()

# -------------------------------- INDICATORS --------------------------------


def zlsma(df, period=50, offset=0, column='close'):
    src = df[column]
    lsma = ta.LINEARREG(src, period, offset)
    lsma2 = ta.LINEARREG(lsma, period, offset)
    eq = lsma - lsma2
    return lsma + eq


def chandelier_exit(df, timeperiod=22, multiplier=3.0, column='close'):
    close = df[column]
    high = df['ha_high']
    low = df['ha_low']
    atr = multiplier * ta.ATR(high, low, close, timeperiod=timeperiod)

    longStop = highest(close, timeperiod) - atr
    longStopPrev = nz(longStop)

    shortStop = lowest(close, timeperiod) + atr
    shortStopPrev = nz(shortStop)

    signal = pd.Series(0, index=df.index)
    signal.loc[close > shortStopPrev] = 1
    signal.loc[close < longStopPrev] = -1

    return signal


def volatility_osc(df, timeperiod=100):
    spike = df['ha_close'] - df['ha_open']
    x = ta.STDDEV(spike, timeperiod)
    y = ta.STDDEV(spike, timeperiod) * -1
    return pd.DataFrame(index=df.index, data={'upper': x, 'lower': y, 'spike': spike})


def calculate_money_flow_volume_series(df):
    mfv = df['volume'] * (2 * df['close'] - df['high'] - df['low']) / \
                                    (df['high'] - df['low'])
    return mfv


def calculate_money_flow_volume(df, n: int = 20):
    return calculate_money_flow_volume_series(df).rolling(n).sum()


def calculate_chaikin_money_flow(df, n: int = 20):
    return calculate_money_flow_volume(df, n) / df['volume'].rolling(n).sum()


def cmf(df, timeperiod=20,):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    ad = np.where((((close == high) & (close == low)) | (high == low)), 0, (
        (2 * close - low - high) / (high - low)) * volume
    )
    signal = pd.Series(ad, index=df.index)

    mf = signal.rolling(timeperiod).sum() / volume.rolling(timeperiod).sum()
    return mf


def supertrend(dataframe: pd.DataFrame, multiplier=3, period=10):
    df = dataframe.copy()

    df['TR'] = ta.TRANGE(df)
    df['ATR'] = ta.SMA(df['TR'], period)

    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)

    # Compute basic upper and lower bands
    df['basic_ub'] = (df['ha_high'] + df['ha_low']) / 2 + multiplier * df['ATR']
    df['basic_lb'] = (df['ha_high'] + df['ha_low']) / 2 - multiplier * df['ATR']

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if (
            (df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1]) or (
                df['ha_close'].iat[i - 1] > df['final_ub'].iat[i - 1])) else \
                df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if (
            (df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1]) or (
                df['ha_close'].iat[i - 1] < df['final_lb'].iat[i - 1])) else  \
            df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if (
                        (df[st].iat[i - 1] == df['final_ub'].iat[i - 1]) and
                        (df['ha_close'].iat[i] <= df['final_ub'].iat[i]))else \
                        df['final_lb'].iat[i] if (
                            (df[st].iat[i - 1] == df['final_ub'].iat[i - 1]) and
                            (df['ha_close'].iat[i] > df['final_ub'].iat[i])) else \
                        df['final_lb'].iat[i] if (
                            (df[st].iat[i - 1] == df['final_lb'].iat[i - 1]) and
                            (df['ha_close'].iat[i] >= df['final_lb'].iat[i])) else \
                        df['final_ub'].iat[i] if (
                            (df[st].iat[i - 1] == df['final_lb'].iat[i - 1]) and
                            (df['ha_close'].iat[i] < df['final_lb'].iat[i])) else 0.00
    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df['ha_close'] < df[st]), 'down',  'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    # df.fillna(0, inplace=True)

    return pd.DataFrame(index=df.index, data={
        'ST': df[st],
        'STX': df[stx]
    })

# POKI indicator


def poki(df):
    close = df['ha_close']
    vol = df['volume']

    atr = ta.ATR(df, timeperiod=14)
    hi, lo, currclose = close, close, close

    prevclose = currclose.shift().fillna(0)
    prevhigh = prevclose + atr
    prevlow = prevclose - atr
    currclose = np.where(hi > prevhigh, hi,
                         np.where(lo < prevlow, lo, prevclose))

    direction = pd.Series(0, index=df.index)
    direction = np.where(currclose > prevclose, 1, np.where(
        currclose < prevclose, -1, direction))

    direction = pd.Series(direction, index=df.index)
    # direction = nz(direction)
    directionIsDown = direction <= 0
    res = vol
    x = np.where(directionIsDown, -res, res)
    # print(direction.to_numpy())
    # print(x)

    # Conditions
    longCond = qtpylib.crossed_above(x, 0)
    shortCond = qtpylib.crossed_below(x, 0)

    # Count your long short conditions for more control with Pyramiding
    sectionLongs = pd.Series(0, index=df.index)
    sectionShorts = pd.Series(0, index=df.index)

    sectionLongs = np.where(longCond, sectionLongs + 1, sectionLongs)
    sectionShorts = np.where(shortCond, sectionShorts + 1, sectionShorts)

    # Pyramiding
    pyrl = 1

    # These check to see your signal and cross references it against the pyramiding settings above
    longCondition = longCond & (sectionLongs <= pyrl)
    shortCondition = shortCond & (sectionShorts <= pyrl)

    return pd.DataFrame(index=df.index, data={
        'long': longCondition,
        'short': shortCondition,
    })


# ---------------

# def Nadaraya_Watson_Envelope(df):
#     input = df['close']
#     src = input.copy()
#     src = src.loc[::-1].reset_index(drop=True)
#     # Settings
#     h = 8
#     r = 8
#     x_0 = 25
#     lag = 2
#     size = len(src)
#     def _kernel_regression(_src, _size, _h):
#         # yhat = [nan] * (x_0 + lag)
#         yhat = []
#         sum_es = []
#         sum_e = 0
#         for i in range(_size - (x_0 + lag)):
#             _currentWeight = 0.
#             _cumulativeWeight = 0.
#             for j in range(i, i + x_0 + lag):
#                 y = _src[j] 
#                 w = math.pow(1 + (math.pow(i-j, 2) / ((math.pow(_h, 2) * 2 * r))), -r)
#                 _currentWeight += (y * w)
#                 _cumulativeWeight += w
#             y2 = _currentWeight / _cumulativeWeight
#             sum_e += abs(src[i] - y2)
#             yhat.append(y2)
#             sum_es.append(sum_e)

#         for i in range((x_0 + lag)):
#             yhat.append(nan)
#             sum_es.append(nan)
#         return yhat, sum_es

#     # Estimations
#     yhat11, sum_es = _kernel_regression(src, size, h)

#     yhat11.reverse()
#     sum_es.reverse()

#     yhat1 = pd.Series(yhat11)
#     mae = sum_es[-1]
#     cross_up = yhat1 + mae
#     cross_dn = yhat1 - mae

#     print(cross_up)

#     return {
#         'up': cross_up,
#         'dn': cross_dn
#     }

def smma(df, timeperiod = 32):
    df['ma'] = ta.SMA(df, timeperiod=timeperiod)
    smma = (df['ma'].shift(1) * (timeperiod - 1) + df['close']) / timeperiod
    return smma


def kernel_regression(input, loop_back = 8):
    src = input.copy()
    src = src.loc[::-1].reset_index(drop=True)
    # Settings
    h = loop_back
    r = 8
    x_0 = 25
    lag = 2
    size = len(src)
    smoothColors = False

    def _kernel_regression(_src, _size, _h):
        yhat = []
        sum_es = []
        sum_e = 0
        for i in range(_size - (x_0 + lag)):
            _currentWeight = 0.
            _cumulativeWeight = 0.
            for j in range(i, i + x_0 + lag):
                y = _src[j] 
                w = math.pow(1 + (math.pow(i-j, 2) / ((math.pow(_h, 2) * 2 * r))), -r)
                _currentWeight += (y * w)
                _cumulativeWeight += w
            y2 = _currentWeight / _cumulativeWeight
            sum_e += abs(src[i] - y2)
            yhat.append(y2)
            sum_es.append(sum_e)

        for i in range((x_0 + lag)):
            yhat.append(nan)
            sum_es.append(nan)
        return yhat, sum_es

    # Estimations
    yhat11, sum_es = _kernel_regression(src, size, h)
    yhat11.reverse()
    return pd.Series(yhat11)


def gaussian_regression(input, loop_back = 8):
    src = input.copy()
    src = src.loc[::-1].reset_index(drop=True)
    # Settings
    h = loop_back
    r = 8
    x_0 = 25
    lag = 2
    size = len(src)
    
    def _gaussian_regression(_src, _size, _h):
        yhat = []
        sum_e = 0
        for i in range(_size - (x_0 + lag)):
            _currentWeight = 0.
            _cumulativeWeight = 0.
            for j in range(i, i + x_0 + lag):
                y = _src[j] 
                w = math.exp(-math.pow(i-j, 2) / (2 * math.pow(_h, 2)))
                _currentWeight += (y * w)
                _cumulativeWeight += w
            y2 = _currentWeight / _cumulativeWeight
            sum_e += abs(src[i] - y2)
            yhat.append(y2)

        for i in range((x_0 + lag)):
            yhat.append(nan)
        return yhat

    # Estimations
    yhat11 = _gaussian_regression(src, size, h)

    yhat11.reverse()
    yhat1 = pd.Series(yhat11)
    return yhat1





def locally_periodic(input, loop_back = 8, _period=24):
    src = input.copy()
    src = src.loc[::-1].reset_index(drop=True)
    # Settings
    h = loop_back
    r = 8
    x_0 = 25
    lag = 2
    size = len(src)
    
    def _locally_periodic(_src, _size, _h):
        yhat = []
        sum_e = 0
        for i in range(_size - (x_0 + lag)):
            _currentWeight = 0.
            _cumulativeWeight = 0.
            for j in range(i, i + x_0 + lag):
                y = _src[j] 
                w = math.exp(-2*math.pow(math.sin(math.pi * (i-j) / _period), 2) / math.pow(_h, 2)) * math.exp(-math.pow(i-j, 2) / (2 * math.pow(_h, 2)))
                _currentWeight += (y * w)
                _cumulativeWeight += w
            y2 = _currentWeight / _cumulativeWeight
            sum_e += abs(src[i] - y2)
            yhat.append(y2)

        for i in range((x_0 + lag)):
            yhat.append(nan)
        return yhat

    yhat11 = _locally_periodic(src, size, h)

    yhat11.reverse()
    yhat1 = pd.Series(yhat11)
    return yhat1