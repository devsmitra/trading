import talib.abstract as ta


def maColor(df, i):
    key = "val" + str(i)
    ma = "ma" + str(i)
    df[key] = df[ma].diff()
    df.loc[(df[key] > 0) & (df[ma] > df['ma100']), key] = 2
    df.loc[(df[key] < 0) & (df[ma] > df['ma100']), key] = -1
    df.loc[(df[key] <= 0) & (df[ma] < df['ma100']), key] = -2
    df.loc[(df[key] >= 0) & (df[ma] < df['ma100']), key] = 1
    df.loc[(df[key] == 0), key] = 0


def madrid(df):
    df["ma100"] = ta.EMA(df, timeperiod=100)
    list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    for i in list:
        df["ma" + str(i)] = ta.EMA(df, timeperiod=i)
        maColor(df, i)
