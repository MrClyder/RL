import collections
import time
import os
import glob
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv

def next_minute(date_str, time_interval):
    ds = datetime.datetime.strptime(date_str, "%H:%M:%S")
    dt = datetime.timedelta(seconds=time_interval)
    return (ds+dt).strftime("%H:%M:%S")

def last_minute(date_str,time_interval):
    ds = datetime.datetime.strptime(date_str, "%H:%M:%S")
    dt = datetime.timedelta(seconds=time_interval)
    return (ds-dt).strftime("%H:%M:%S")

def next_minute_1(date_str):
    ds = datetime.datetime.strptime(date_str, "%H:%M:%S")
    dt = datetime.timedelta(seconds=1)
    return (ds+dt).strftime("%H:%M:%S")

def Sum(a):
    s = 0
    for i in range(1,a+1):
        s += i
    return s

def Ma(data, time_lenth):
    ma_list = []
    df = collections.deque([], maxlen=time_lenth)
    for i in range(len(data)):
        df.append(data[i])
        ma = np.mean(data)
        ma_list.append(ma)
    return ma_list

def Ema(data, time_lenth):
    ema = []
    ema.append(data[0])
    for i in range(1, len(data)):
        ema_data = ema[i-1] * (2/(time_lenth+1)) + (1-(2/(time_lenth+1))) * data[i]
        ema.append(ema_data)
    return ema

def Macd(data, time_long, time_short, time_lenth):
    Diff = []
    macd = []
    dea_list = [0]
    ema_long = Ema(data, time_long)
    ema_short = Ema(data, time_short)
    for i in range(len(ema_long)):
        Diff.append(ema_short[i] - ema_long[i])
    macd.append(Diff[0])
    for i in range(1, len(Diff)):
        DEA = macd[i-1] * (1-(2/(time_lenth+1))) + Diff[i] * (2/(time_lenth+1))
        dea_list.append(DEA)
        macd_data = (Diff[i] - DEA) ** 2
        macd.append(macd_data)
    return macd, Diff, dea_list

def Boll(data, time_lenth, k):
    ave_list = []
    std_list = []
    up_line = []
    down_line = []
    df = collections.deque([], maxlen=time_lenth)
    for i in range(len(data)):
        df.append(data[i])
        ave_list.append(np.mean(df))
        std_list.append(np.std(df))
    for i in range(len(ave_list)):
        up_line.append(ave_list[i] + k * std_list[i])
        down_line.append(ave_list[i] - k * std_list[i])

    return ave_list, up_line, down_line

def Kdj(data, time_lenth):
    rsv = Rsv(data, time_lenth)
    K_list = [50]
    D_list = [50]
    J_list = []
    for i in range(1, len(rsv)):
        k = K_list[i-1] * (2/3) + (1/3) * rsv[i]
        K_list.append(k)
    for i in range(1, len(rsv)):
        d = D_list[i-1] * (2/3) + (1/3) * K_list[i]
        D_list.append(d)
    for i in range(len(rsv)):
        j = 3 * K_list[i] - 2 * D_list[i]
        J_list.append(j)
    return K_list, D_list, J_list

def Rsv(data, time_lenth):
    rsv_list = []
    df = collections.deque([],maxlen=time_lenth)
    for i in range(len(data)):
        df.append(data[i])
        max = np.max(df)
        min = np.min(df)
        rsv_list.append((data[i] - max) / (max - min))
    return rsv_list

def Rsi(data, time_lenth):
    df = collections.deque([], maxlen=time_lenth)
    rsi_list = []
    for i in range(len(data)):
        rise = 0
        down = 0
        df.append(data[i])
        for j in range(len(df)-1):
            if df[j] < df[j+1]:
                rise += (df[j+1] - df[j])
            else:
                down += (df[j] - df[j+1])
        rs = rise/(rise+down)
        rsi = (rs/(1+rs)) * 100
        rsi_list.append(rsi)
    return rsi_list

def Cci(data, time_lenth):
    ma_list = Ma(data, time_lenth)
    tp_list = []
    md_list = []
    cci_list = []
    k = 0.015
    md = collections.deque([], maxlen=time_lenth)
    df = collections.deque([], maxlen=time_lenth)
    for i in range(len(data)):
        df.append(data[i])
        tp = (np.max(df)+np.min(df)+data[i]) / 3
        tp_list.append(tp)
    for i in range(len(data)):
        md.append(np.abs(ma_list[i] - data[i]))
        for j in range(len(md)):
            md_list.append(np.mean(md))
    for i in range(len(data)):
        cci = (tp_list[i]-ma_list[i])/md_list[i]/k
        cci_list.append(cci)
    return cci_list

def data_fetch(filename,save_filename):
    """
        filename为数据路径
        从原始数据中提取trade数据，目前提取的为 price,amount,ts,可自行增减
    """

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        price_list = []
        ts_list = []
        amount_list = []
        direction_list = []
        for df in reader:
            data = eval(df[0])
            if data.get('ch') == 'market.ETH-USDT.trade.detail':
                for j in range(len(data['tick']['data'])):
                    price = data['tick']['data'][j].get('price')
                    ts = data['tick']['data'][j].get('ts')
                    amount = data['tick']['data'][j].get('quantity')
                    direction = data['tick']['data'][j].get('direction')
                    price_list.append(price)
                    ts_list.append(ts)
                    amount_list.append(amount)
                    direction_list.append(direction)
        prep_data = pd.DataFrame({'price': price_list, 'ts': ts_list, 'amount': amount_list, 'direction': direction_list})
        data.to_csv(save_filename)
        return prep_data

def data_aggregation_0_1s(filename, time_interval):
    """
        filename为数据路径
        仅仅适用于0.1秒级的数据处理
        输入的数据需包含price,ts数据，如没有volume或有其他的数据，需在后面cnames增减
    """
    cnames = ['price', 'ts', 'volume', 'drection']
    df = pd.read_csv(filename, header=0, names=cnames)
    ts = df['ts'].values
    time = []
    for i in ts:
        t = datetime.datetime.fromtimestamp(i / 1000.0).isoformat()
        time.append(t)
    df['time'] = time
    df['tradingday'], df['updatetime'] = df['time'].str.split('[ T]').str
    df['updateminute'], df['millisecond'] = df['updatetime'].str.split('[.]').str
    df = df.drop(columns=['time', 'updatetime'], axis=1)
    df = df.dropna()
    # print(df)
    df1 = df.copy()
    df1.reset_index(drop=True, inplace=True)
    for i in range(len(df1)):
        milli = df1['millisecond'][i]
        if milli[0] == '9' and milli != '900000':
            df1['updateminute'][i] = next_minute_1(df1['updateminute'][i])
            df1['millisecond'][i] = '000000'
        elif milli[-5:] != '00000':
            a = int(milli[0]) + 1
            a = str(a)
            df1['millisecond'][i] = a + '00000'
    df1['date'] = df1[df1.columns[-2:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    grouped = df1.groupby(['date'])
    dfnew = grouped.agg({'ts': 'last', 'price': 'last', 'volume': 'last', \
                         'tradingday': 'last', 'updateminute': 'last', 'millisecond': 'last', 'date': 'last', \
                         'direction': 'last'})

    dfnew.reset_index(drop=True, inplace=True)
    dfnew['date'] = pd.to_datetime(dfnew['date'], format='%H:%M:%S %f').dt.time
    # print(dfnew)
    start = df1['updateminute'][0] + '.' + df1['millisecond'][0]
    end = df1['updateminute'].iloc[-1] + '.' + df1['millisecond'].iloc[-1]

    dates = pd.date_range(start=start, end=end, freq=time_interval).time
    dfnew = dfnew.set_index('date').reindex(dates).reset_index().reindex(columns=dfnew.columns)
    cols = dfnew.columns
    dfnew[cols] = dfnew[cols].ffill()
    print(dfnew)
    return dfnew

def data_aggregation_1s(filename, time_interval):
    """
        filename为数据路径，time_interval为时间间隔，单位为1s
        适用于整秒级的数据处理
        输入的数据需包含price,ts数据，如没有volume或有其他的数据，需在后面cnames增减
    """
    cnames = ['price', 'ts', 'volume', 'direction']
    df = pd.read_csv(filename, header=0, names=cnames)
    ts = df['ts'].values
    time = []
    for i in ts:
        t = datetime.datetime.fromtimestamp(i // 1000.0).isoformat()
        time.append(t)
    df['time'] = time
    df['tradingday'], df['updatetime'] = df['time'].str.split('[ T]').str
    df = df.drop(columns=['time'], axis=1)
    df = df.dropna()

    grouped = df.groupby(['updatetime'])
    df = grouped.agg({'ts': 'last', 'price': 'last', 'volume': 'last', \
                         'tradingday': 'last', 'updatetime': 'last', \
                         'direction': 'last'})
    # print(df)
    df.reset_index(drop=True, inplace=True)
    df['date'] = pd.to_datetime(df['updatetime'], format='%H:%M:%S').dt.time
    start = df['updatetime'][0]
    # start = next_minute(start, int(time_interval[:-1]))
    end = df['updatetime'].iloc[-1]
    dates = pd.date_range(start=start, end=end, freq=time_interval).time
    dfnew = df.set_index('date').reindex(dates).reset_index().reindex(columns=df.columns)
    dfnew = dfnew.drop(columns=['updatetime'], axis=1)
    # dfnew = dfnew.drop(index=[0]).reset_index()
    cols = dfnew.columns
    dfnew[cols] = dfnew[cols].ffill()
    print(dfnew)
    return dfnew

def data_sort(filename, time_interval, time_span, slide_move):
    """
        filename为数据路径，time_span为观测时长，slide_move为预测时长,
        time_interval为数据时间间隔
    """
    # if k == '1s':
    #     fun = data_aggregation_1s
    # else:
    #     fun = data_aggregation_0_1s
    # df = fun(filename, time_interval)
    df = pd.read_csv(filename)
    df = np.array(df['price'])
    results = []
    time_interval = int(time_interval[:-1])
    # print(time_interval)
    for i in range(len(df) // ((time_span + slide_move) // time_interval)):
        results.append([df[j] for j in range(i * (time_span + slide_move), (i + 1) * (time_span + slide_move))])
    results = pd.DataFrame(results)
    return results

def mean_ratio(data, time_span, slide_move):
    """
        data需为经过处理的数据，time_span为观测时长，slide_move为预测时长
        预测时间内的均值与观测时间跨度内最后一点的比值
    """
    ratio = []
    k = len(data)
    data = data.T
    for i in range(k):
        last_price = data[i][time_span-1]
        mean = np.mean(data[i][-slide_move:])
        ratio.append((last_price - mean)/last_price)
    data = data.T
    data['ratio'] = ratio
    return data


def DOM(filename, time_interval, save_filename):

    """
    buy_df,sell_df 分别是带有price，ts，amount的交易数据
    先将时间转化为s,将价格精度保留小数点1位，然后根据不同的时间间隔取数据
    接着根据价格将买卖表格拼接在一起生成新的表格(sub_df)
    最后将生成的数据存为csv

    """

    df = pd.read_csv(filename, index_col=0)
    buy = df.loc[df['direction'] == 'buy']
    sell = df.loc[df['direction'] == 'sell']
    buy = buy[['price', 'ts', 'amount']].reset_index()
    sell = sell[['price', 'ts', 'amount']].reset_index()
    sell.price = round(sell.price, 1)
    buy.price = round(buy.price, 1)
    length = int((max(sell.ts) - min(sell.ts)) / time_interval)

    for j in range(length):
        list = {}
        start_time = sell.ts[0] + j * time_interval
        sub_sell = sell[(sell['ts'] <= (start_time + time_interval)) & (sell['ts'] >= start_time)]
        sub_sell = sub_sell.groupby("price").amount.sum()
        sub_buy = buy[(buy['ts'] <= (start_time + time_interval)) & (buy['ts'] >= start_time)]
        sub_buy = sub_buy.groupby("price").amount.sum()
        sub_df = (pd.merge(sub_sell, sub_buy, on='price', how='outer',
                           suffixes=('_sell', '_buy'))).fillna(0)
        sub_df = (sub_df.sort_values(by="price", ascending=False)).reset_index()
        for i in range(len(sub_df)):
            list['ts'] = sell.ts[0] + j * time_interval
            list[sub_df['price'][i]] = {'sell_amount': sub_df['amount_sell'][i], 'buy_amount': sub_df['amount_buy'][i]}
        with open(save_filename, 'a') as f:
            write = csv.writer(f)
            write.writerow([list])
    return sub_df


def imbalance_occur(df, time_gap, imbalance_rate):

    """
    buy_df,sell_df 分别是带有price，ts，amount的交易数据
    time_gap（float,单位：秒）取决于目的和该币种的交易量，eth这种交易量比较少的可以取60s计算一次
    imbalance_rate（float，比值）

    输出结果是一个5列的数据表，含有：price(成交价格，单位：美元)，
                                amount_sell/amount_buy(主动方成交量)
                                r(sell/buy的比值)
                                count(超过设定的imbalance的阈值点的个数)

    """
    df = pd.read_csv(df)
    buy = df.loc[df['direction'] == 'buy']
    buy = buy[['price', 'ts', 'amount']].reset_index()
    sell = df.loc[df['direction'] == 'sell']
    sell = sell[['price', 'ts', 'amount']].reset_index()

    sell.price = round(sell.price, 1)
    buy.price = round(buy.price, 1)

    largest_r = pd.DataFrame()
    length = int((max(sell.ts) - min(sell.ts)) / time_gap)

    for j in range(length):
        start_time = sell.ts[0] + j * time_gap
        sub_sell = sell[(sell['ts'] <= (start_time + time_gap)) & (sell['ts'] >= start_time)]
        sub_sell = sub_sell.groupby("price").amount.sum()
        sub_buy = buy[(buy['ts'] <= (start_time + time_gap)) & (buy['ts'] >= start_time)]
        sub_buy = sub_buy.groupby("price").amount.sum()
        sub_df = (pd.merge(sub_sell, sub_buy, on='price', how='outer', suffixes=('_sell', '_buy'))).fillna(0)
        sub_df = (sub_df.sort_values(by="price", ascending=False)).reset_index()
        r = [0]
        for i in range(len(sub_df) - 1):
            if sub_df.amount_sell.iloc[i] == 0:
                r.append(float('0.2'))
            else:
                r.append(sub_df.amount_buy.iloc[i + 1] / sub_df.amount_sell.iloc[i])
        sub_df['r'] = r
        count = sum(i > imbalance_rate for i in r)
        sub_df['count'] = [count] * len(sub_df)
        max_imbalance = sub_df[sub_df.r == sub_df.r.max()]

        min_imbalance = sub_df[sub_df.r == sub_df.r.min()]

        if len(max_imbalance) == 1:
            largest_r = largest_r.append(max_imbalance)
        else:
            pass
    return largest_r


#############################################################################

def get_POC_closing_signal(df, time_gap):
    df = pd.read_csv(df)
    buy = df[df['direction'] == 'buy']
    buy = buy[['price', 'ts', 'amount']].reset_index()
    sell = df[df['direction'] == 'sell']
    sell = sell[['price', 'ts', 'amount']].reset_index()

    sell.price = round(sell.price, 1)
    buy.price = round(buy.price, 1)
    length = int((max(sell.ts) - min(sell.ts)) / time_gap)
    # print(length)
    POC = []
    ts_list = []
    day_list = []
    date_list = []
    closing = []
    signal = []
    for j in range(length+1):

        a = sell['ts'][0] + j * time_gap

        sub_sell = sell[(sell['ts'] <= (a + time_gap)) & (sell['ts'] >= a)]
        ts = a + time_gap
        # t = datetime.datetime.fromtimestamp(ts // 1000.0).isoformat()
        # day, date = t.str.split('[ T]').str
        sell_close = sub_sell.tail(1)
        sub_sell = sub_sell.groupby("price").amount.sum()
        sub_buy = buy[(buy['ts'] <= (a + time_gap)) & (buy['ts'] >= a)]

        if (sub_buy.empty == True) or (sub_sell.empty == True):
            poc = None
            Close = None
            signal.append('Unknown')
        else:
            buy_close = sub_buy.tail(1)
            sub_buy = sub_buy.groupby("price").amount.sum()
            sub_df = (pd.merge(sub_sell, sub_buy, on='price', how='outer', suffixes=('_sell', '_buy'))).fillna(0)
            sub_df = (sub_df.sort_values(by="price", ascending=False)).reset_index()
            sub_df['accum_amount'] = sub_df['amount_buy'] + sub_df['amount_sell']
            # POC.append(sub_df[sub_df.accum_amount == sub_df.accum_amount.max()].price.values[0])
            poc = sub_df[sub_df.accum_amount == sub_df.accum_amount.max()].price.values[0]
            if float(buy_close.ts) > float(sell_close.ts):
                Close = float(buy_close.price)
            else:
                Close = float(sell_close.price)
            if poc > Close:
                signal.append('bearish')
            elif poc < Close:
                signal.append('bullish')
            else:
                signal.append('Unknown')
        closing.append(Close)
        ts_list.append(ts)
        # day_list.append(day)
        # date_list.append(date)
        POC.append(poc)
    POCsignal = pd.concat([pd.DataFrame(ts_list, columns=['ts']),
                        # pd.DataFrame(day_list, columns=['day']),
                        # pd.DataFrame(date_list, columns=['date']),
                        pd.DataFrame(POC, columns=['POC']),
                        pd.DataFrame(closing, columns=['close']),
                        pd.DataFrame(signal, columns=['signal'])], axis=1)
    ts = POCsignal['ts'].values
    time = []
    for i in ts:
        t = datetime.datetime.fromtimestamp(i // 1000.0).isoformat()
        time.append(t)
    POCsignal['time'] = time
    POCsignal['day'], POCsignal['updatetime'] = POCsignal['time'].str.split('[ T]').str
    POCsignal = POCsignal.drop(columns=['time'], axis=1)
    POCsignal = POCsignal.fillna(method='ffill')

    return POCsignal

##############################Delta########################################

def Delta_Open_Close_high(df, time_gap):

    """
    buy,sell分别是带有price，ts，amount的交易数据
    time_gap（float,单位：毫秒）取决于目的和该币种的交易量，eth这种交易量比较少的可以取60s计算一次

    输出结果是一个5列的数据表，含有：delta(交易量差值)，
                                open/close/highest(单位：美元)
                                signal(信号，根据书里的内容进行判断)

    """
    df = pd.read_csv(df)
    buy = df[df['direction'] == 'buy']
    buy = buy[['price', 'ts', 'amount']].reset_index()
    sell = df[df['direction'] == 'sell']
    sell = sell[['price', 'ts', 'amount']].reset_index()

    length = int((max(sell.ts) - min(sell.ts)) / time_gap)
    # print(length)
    Delta = []
    ts_list = []
    Open_price = []
    Close_price = []
    highest = []

    for j in range(length+1):
        start_time = sell.ts[0] + j * time_gap
        sub_sell = sell[(sell['ts'] <= (start_time + time_gap)) & (sell['ts'] >= start_time)]
        ts = start_time + time_gap
        sell_open = sub_sell.head(1)
        sell_close = sub_sell.tail(1)
        sub_sell = sub_sell.groupby("price").amount.sum()

        sub_buy = buy[(buy['ts'] <= (start_time + time_gap)) & (buy['ts'] >= start_time)]
        buy_open = sub_buy.head(1)
        buy_close = sub_buy.tail(1)
        sub_buy = sub_buy.groupby("price").amount.sum()

        if (sub_buy.empty == True) or (sub_sell.empty == True):
            Delta.append(None)
            Close = None
            Open = None
            high = None
        else:
            sub_df = (pd.merge(sub_sell, sub_buy, on='price', how='outer', suffixes=('_sell', '_buy'))).fillna(0)
            sub_df = (sub_df.sort_values(by="price", ascending=False)).reset_index()
            Delta.append(sub_df.amount_buy.sum() - sub_df.amount_sell.sum())
            high = sub_df.price.max()
            if float(buy_open.ts) > float(sell_open.ts):
                Open = float(sell_open.price)
            else:
                Open = float(buy_open.price)

            if float(buy_close.ts) > float(sell_close.ts):
                Close = float(buy_close.price)
            else:
                Close = float(sell_close.price)

        Open_price.append(Open)
        Close_price.append(Close)
        highest.append(high)
        ts_list.append(ts)
    Delta_open_close = pd.concat([pd.DataFrame(ts_list, columns=['ts']),
                                pd.DataFrame(Delta, columns=['delta']),
                                pd.DataFrame(Close_price, columns=['close']),
                                pd.DataFrame(Open_price, columns=['open']),
                                pd.DataFrame(highest, columns=['highest'])], axis=1)
    signal = []
    for i in range(len(Delta_open_close)):
        if ((Delta_open_close.delta[i] > 0) and (Delta_open_close.close[i] > Delta_open_close.open[i])):
            signal.append('bullish')
        elif ((i > 2) and (Delta_open_close.delta[i] < 0) and (Delta_open_close.close[i] < Delta_open_close.open[i])
              and (Delta_open_close.highest[i] > Delta_open_close.highest[i - 1])
              and (Delta_open_close.highest[i] > Delta_open_close.highest[i - 2])):
            signal.append('bearish')
        else:
            signal.append('unknow')
    Delta_open_close['signal'] = signal
    Delta_open_close = Delta_open_close.fillna(method='ffill')

    ts = Delta_open_close['ts'].values
    time = []
    for i in ts:
        t = datetime.datetime.fromtimestamp(i // 1000.0).isoformat()
        time.append(t)
    Delta_open_close['time'] = time
    Delta_open_close['day'], Delta_open_close['updatetime'] = Delta_open_close['time'].str.split('[ T]').str
    Delta_open_close = Delta_open_close.drop(columns=['time'], axis=1)
    Delta_open_close = Delta_open_close.fillna(method='ffill')

    return Delta_open_close


##############################min/max delta########################################

def Delta_table(df, time_gap):

    """
    df分别是带有price，ts，amount,direction的交易数据
    time_gap（float,单位：毫秒）取决于目的和该币种的交易量，eth这种交易量比较少的可以取60s计算一次

    输出结果是一个3列的数据表，含有：delta(交易量差值)，
                                min_delta/max_delta(交易量差值)

    """
    df = pd.read_csv(df)
    length = int((max(df.ts) - min(df.ts)) / time_gap)
    print(length)
    min_delta = []
    max_delta = []
    ts_list = []
    delta = []

    for j in range(length):
        start_time = df.ts[0] + j * time_gap
        sub_df = df[(df['ts'] <= (start_time + time_gap)) & (df['ts'] >= start_time)]
        ts = start_time + time_gap
        cum_Delta = []
        Delta = []
        if sub_df.empty == True:
            min_delta.append(None)
            max_delta.append(None)
            delta.append(None)
        else:
            for i in range(len(sub_df)):
                if sub_df.direction.iloc[i] == 'sell':
                    Delta.append(float(sub_df.amount.iloc[i]) * (-1))
                    cum_Delta.append(sum(Delta))
                else:
                    Delta.append(float(sub_df.amount.iloc[i]))
                    cum_Delta.append(sum(Delta))
        min_delta.append(min(cum_Delta))
        max_delta.append(max(cum_Delta))
        delta.append(cum_Delta[-1])
        ts_list.append(ts)
    Delta_df = pd.concat([pd.DataFrame(ts_list, columns=['ts']),
                        pd.DataFrame(delta, columns=['delta']),
                        pd.DataFrame(min_delta, columns=['min_delta']),
                        pd.DataFrame(max_delta, columns=['max_delta'])], axis=1)
    Delta_df = Delta_df.fillna(method='ffill')

    ts = Delta_df['ts'].values
    time = []
    for i in ts:
        t = datetime.datetime.fromtimestamp(i // 1000.0).isoformat()
        time.append(t)
    Delta_df['time'] = time
    Delta_df['day'], Delta_df['updatetime'] = Delta_df['time'].str.split('[ T]').str
    Delta_df = Delta_df.drop(columns=['time'], axis=1)
    Delta_df = Delta_df.fillna(method='ffill')

    return Delta_df

def factor_data(source_path, time_interval, save_path):
    data_list = sorted(glob.glob(source_path + '/*.csv'))
    for i in range(len(data_list)):
        contract = data_list[i]
        filename = str(contract.split('\\')[-1].split('_')[5])
        POC = get_POC_closing_signal(contract, time_interval)
        Delta = Delta_Open_Close_high(contract, time_interval)
        df = pd.DataFrame(
            {'ts': Delta['ts'], 'day': Delta['day'], 'updatetime': Delta['updatetime'], 'price': Delta['close'],
            'POC': POC['POC'], 'delta': Delta['delta'], 'POC_signal': POC['signal'], 'Delta_signal': Delta['signal']})
        df.to_csv(save_path + 'eth_swap_' + filename + '_{}s_data.csv'.format(time_interval//1000))

def merge_factor(filename_list,time_interval,observe_time,forecast_time,factor):
    data = factor_data(filename_list[0], time_interval)
    data = np.array(data[factor])
    results = []
    for i in range(len(data) // ((observe_time + forecast_time) // time_interval)):
        results.append([data[j] for j in range(i * ((observe_time + forecast_time)//time_interval), (i + 1) * ((observe_time + forecast_time)//time_interval))])
    results = pd.DataFrame(results)
    for i in range(1, len(filename_list)):
        result = []
        df = factor_data(filename_list[i], time_interval)
        df = np.array(df[factor])
        for k in range(len(df) // ((observe_time + forecast_time) // time_interval)):
            result.append([df[j] for j in range(i * ((observe_time + forecast_time)//time_interval), (i + 1) * ((observe_time + forecast_time)//time_interval))])
        result = pd.DataFrame(result)
        results = pd.concat([results, result]).reset_index()
        results = results.drop(columns=['index'], axis=1)
    return results
