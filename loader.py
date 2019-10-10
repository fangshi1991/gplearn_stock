# encoding:utf-8
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import datetime
from collections import OrderedDict

import logbook

import constants

import pandas as pd
#from pandas_datareader import DataReader
import pytz

from six import iteritems
from six.moves.urllib_error import HTTPError

#from . benchmarks import get_benchmark_returns
from mongodb import LoadDataCVS
import treasuries, treasuries_can


logger = logbook.Logger('Loader')

# Mapping from index symbol to appropriate bond data
INDEX_MAPPING = {
    '^GSPC':
    (treasuries, 'treasury_curves.csv', 'www.federalreserve.gov'),
    '^GSPTSE':
    (treasuries_can, 'treasury_curves_can.csv', 'bankofcanada.ca'),
    '^FTSE':  # use US treasuries until UK bonds implemented
    (treasuries, 'treasury_curves.csv', 'www.federalreserve.gov'),
}

ONE_HOUR = pd.Timedelta(hours=1)


def last_modified_time(path):
    """
    Get the last modified time of path as a Timestamp.
    """
    return pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')


def get_data_filepath(name):
    """
    Returns a handle to data file.

    Creates containing directory, if needed.
    """
    dr = data_root()

    if not os.path.exists(dr):
        os.makedirs(dr)

    return os.path.join(dr, name)


def get_cache_filepath(name):
    cr = cache_root()
    if not os.path.exists(cr):
        os.makedirs(cr)

    return os.path.join(cr, name)


def get_benchmark_filename(symbol):
    return "%s_benchmark.csv" % symbol


def has_data_for_dates(series_or_df, first_date, last_date):
    """
    Does `series_or_df` have data on or before first_date and on or after
    last_date?
    """
    dts = series_or_df.index
    if not isinstance(dts, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex, but got %s." % type(dts))
    first, last = dts[[0, -1]]
    return (first <= first_date) and (last >= last_date)


def load_market_data(trading_day,
                     trading_days,
                     bm_symbol='000001'):
    """
    Load benchmark returns and treasury yield curves for the given calendar and
    benchmark symbol.

    Benchmarks are downloaded as a Series from Yahoo Finance.  Treasury curves
    are US Treasury Bond rates and are downloaded from 'www.federalreserve.gov'
    by default.  For Canadian exchanges, a loader for Canadian bonds from the
    Bank of Canada is also available.

    Results downloaded from the internet are cached in
    ~/.zipline/data. Subsequent loads will attempt to read from the cached
    files before falling back to redownload.

    Parameters
    ----------
    trading_day : pandas.CustomBusinessDay, optional
        A trading_day used to determine the latest day for which we
        expect to have data.  Defaults to an NYSE trading day.
    trading_days : pd.DatetimeIndex, optional
        A calendar of trading days.  Also used for determining what cached
        dates we should expect to have cached. Defaults to the NYSE calendar.
    bm_symbol : str, optional
        Symbol for the benchmark index to load.  Defaults to '^GSPC', the Yahoo
        ticker for the S&P 500.

    Returns
    -------
    (benchmark_returns, treasury_curves) : (pd.Series, pd.DataFrame)

    Notes
    -----

    Both return values are DatetimeIndexed with values dated to midnight in UTC
    of each stored date.  The columns of `treasury_curves` are:

    '1month', '3month', '6month',
    '1year','2year','3year','5year','7year','10year','20year','30year'
    #为给定的日历和基准符号加载基准回报和国债收益率曲线。基准测试从Yahoo Finance下载为系列。
    #资金曲线是美国国债利率，默认情况下从'www.federalreserve.gov'下载。
    #对于加拿大交易所，也可以使用加拿大银行的加拿大债券装载机。

    #从互联网下载的结果将被缓存
    #〜/ .zipline /数据。后续加载将尝试从缓存中读取
    #文件在退回重新下载之前。

    #参数
     ----------
    trading_day：pandas.CustomBusinessDay，可选
    交易日用于确定我们的最新日期
    期待有数据。默认为纽约证券交易所交易日。
    trading_days：pd.DatetimeIndex，可选
    交易日的日历。还用于确定缓存的内容
    我们应该期望缓存的日期。默认为纽约证券交易所日历。
    bm_symbol：str，可选
    要加载的基准索引的符号。默认为'^ GSPC'，雅虎
    标准普尔500指数的股票代码。
    
    返回
    -------
    （benchmark_returns，treasury_curves）:( pd.Series，pd.DataFrame）
    
    笔记
    """
    first_date = trading_days[0]
    now = pd.Timestamp.utcnow()

    # We expect to have benchmark and treasury data that's current up until
    # **two** full trading days prior to the most recently completed trading
    # day.
    # Example:
    # On Thu Oct 22 2015, the previous completed trading day is Wed Oct 21.
    # However, data for Oct 21 doesn't become available until the early morning
    # hours of Oct 22.  This means that there are times on the 22nd at which we
    # cannot reasonably expect to have data for the 21st available.  To be
    # conservative, we instead expect that at any time on the 22nd, we can
    # download data for Tuesday the 20th, which is two full trading days prior
    # to the date on which we're running a test.
    # We'll attempt to download new data if the latest entry in our cache is
    # before this date.
    # 我们预计基准和财务数据将持续到最近完成交易日前的两个**完整交易日
    # 例：
    # 2015年10月22日星期四，之前完成的交易日为10月21日星期三。
    # 但是，10月21日的数据直到清晨才可用
    # 10月22日的小时数。这意味着我们22日有时间
    # 不能合理地期望有21日的数据可用。 保守一点，我们反而希望在22日的任何时候，我们都可以
    # 下载20日星期二的数据，这是两个完整的交易日
    # 到我们正在进行测试的日期。 如果缓存中的最新条目在此日期之前，我们将尝试下载新数据。
    #print trading_days.get_loc(now, method='ffill')
    #print trading_days[-1]
    #last_date = trading_days[trading_days.get_loc(now, method='ffill') - 2]
    #print last_date
    last_date = trading_days[-1]
    #raw_input()

    br = ensure_benchmark_data(
        bm_symbol,
        first_date,
        last_date,
        now,
        # We need the trading_day to figure out the close prior to the first
        # date so that we can compute returns for the first date.
        trading_day,
    )

    tc = ensure_treasury_data(
        bm_symbol,
        first_date,
        last_date,
        now,
    )

    benchmark_returns = br[br.index.slice_indexer(first_date, last_date)]
    treasury_curves = tc[tc.index.slice_indexer(first_date, last_date)]
    return benchmark_returns, treasury_curves


def ensure_benchmark_data(symbol, first_date, last_date, now, trading_day):
    """
    Ensure we have benchmark data for `symbol` from `first_date` to `last_date`

    Parameters
    ----------
    symbol : str
        The symbol for the benchmark to load.
    first_date : pd.Timestamp
        First required date for the cache.
    last_date : pd.Timestamp
        Last required date for the cache.
    now : pd.Timestamp
        The current time.  This is used to prevent repeated attempts to
        re-download data that isn't available due to scheduling quirks or other
        failures.
    trading_day : pd.CustomBusinessDay
        A trading day delta.  Used to find the day before first_date so we can
        get the close of the day prior to first_date.

    We attempt to download data unless we already have data stored at the data
    cache for `symbol` whose first entry is before or on `first_date` and whose
    last entry is on or after `last_date`.

    If we perform a download and the cache criteria are not satisfied, we wait
    at least one hour before attempting a redownload.  This is determined by
    comparing the current time to the result of os.path.getmtime on the cache
    path.
    """

    # If the path does not exist, it means the first download has not happened
    # yet, so don't try to read from 'path'.

    try:
        data = get_benchmark_returns(
            symbol,
            first_date - trading_day,
            last_date,
        )
    except (OSError, IOError, HTTPError):
        logger.exception('failed to cache the new benchmark returns')
    if not has_data_for_dates(data, first_date, last_date):
        logger.warn("Still don't have expected data after redownload!")
    return data


def ensure_treasury_data(bm_symbol, first_date, last_date, now):
    """
    Ensure we have treasury data from treasury module associated with
    `bm_symbol`.

    Parameters
    ----------
    bm_symbol : str
        Benchmark symbol for which we're loading associated treasury curves.
    first_date : pd.Timestamp
        First date required to be in the cache.
    last_date : pd.Timestamp
        Last date required to be in the cache.
    now : pd.Timestamp
        The current time.  This is used to prevent repeated attempts to
        re-download data that isn't available due to scheduling quirks or other
        failures.

    We attempt to download data unless we already have data stored in the cache
    for `module_name` whose first entry is before or on `first_date` and whose
    last entry is on or after `last_date`.

    If we perform a download and the cache criteria are not satisfied, we wait
    at least one hour before attempting a redownload.  This is determined by
    comparing the current time to the result of os.path.getmtime on the cache
    path.
    """
    # loader_module, filename, source = INDEX_MAPPING.get(
    #     bm_symbol, INDEX_MAPPING['^GSPC']
    # )
    # first_date = max(first_date, loader_module.earliest_possible_date())
    # path = get_data_filepath(filename)

    # # If the path does not exist, it means the first download has not happened
    # # yet, so don't try to read from 'path'.
    # if os.path.exists(path):
    #     try:
    #         data = pd.DataFrame.from_csv(path).tz_localize('UTC')
    #         if has_data_for_dates(data, first_date, last_date):
    #             return data

    #         # Don't re-download if we've successfully downloaded and written a
    #         # file in the last hour.
    #         last_download_time = last_modified_time(path)
    #         if (now - last_download_time) <= ONE_HOUR:
    #             logger.warn(
    #                 "Refusing to download new treasury data because a "
    #                 "download succeeded at %s." % last_download_time
    #             )
    #             return data

    #     except (OSError, IOError, ValueError) as e:
    #         # These can all be raised by various versions of pandas on various
    #         # classes of malformed input.  Treat them all as cache misses.
    #         logger.info(
    #             "Loading data for {path} failed with error [{error}].".format(
    #                 path=path, error=e,
    #             )
    #         )

    # try:
    #     data = loader_module.get_treasury_data(first_date, last_date)
    #     data.to_csv(path)
    # except (OSError, IOError, HTTPError):
    #     logger.exception('failed to cache treasury data')
    # if not has_data_for_dates(data, first_date, last_date):
    #     logger.warn("Still don't have expected data after redownload!")
    l=LoadDataCVS(constants.IP,constants.PORT)
    l.Conn()
    data=l.read_treasure_from_mongodb(first_date, last_date)
    l.Close()
    return data


#提取分钟数据，这里我们提取
def load_day_data(indexes=None,stockList=None,start=None,end=None,adjusted=True,rolling_count= 10):
    # 
    """
    load stocks from Mongo
    """
    assert indexes is not None or stockList is not None, """
must specify stockList or indexes"""
    #对日期进行改造，提取的数据日期应该高于多于开始日期一个月，这样对于原数据有缓冲作用
    start_time = pd.Timestamp(start,tz='UTC')
    end_time = pd.Timestamp(end,tz='UTC')

    if start is None:
        start = "1990-01-01"

    if start is not None and end is not None:
        startdate = datetime.datetime.strptime(start, "%Y-%m-%d")
        enddate=datetime.datetime.strptime(end, "%Y-%m-%d")
        assert startdate < enddate, "start date is later than end date."

    data = OrderedDict()
    start = (datetime.datetime.strptime(start,'%Y-%m-%d')-datetime.timedelta(days=rolling_count+1)).strftime('%Y-%m-%d')

    l=LoadDataCVS(constants.IP,constants.PORT)
    l.Conn()

    if stockList=="hs300" or stockList=="zz500" or stockList=="sz50" or stockList=="all":
        stocks=l.getstocklist(stockList)
    else:
        stocks=stockList
    
    #print stocks

    if stocks is not None:
        for stock in stocks:
            stkd= l.getstockdaily(stock,start,end)
            if not adjusted:   
                data[stock] = stkd
            else:
                adj_cols = ['open', 'high', 'low', 'close']
                ratio = stkd['price']/stkd['close']
                ratio_filtered = ratio.fillna(0).values
                for col in adj_cols:
                    stkd[col] *= ratio_filtered
                data[stock] = stkd
        return [data,start_time,end_time]        
            
    
    if indexes is not None:
        stkd= l.getindexdaily(indexes,start,end)
        data[indexes] = stkd
        return data
        '''
        for name, ticker in items(indexes):
            print (name,ticker)
            logger.info('Loading index: {} ({})'.format(name, ticker))
            stkd= l.getindexdaily(indexes,start,end)
            data[name] = stkd
        return data 
        '''
        
        
        
    '''
    #['open','high','low','close','volume','price','change',"code"]
    print (data)
    panel = pd.Panel(data)
    panel.minor_axis = ['open', 'high', 'low', 'close', 'volume', 'price','change','code']
    panel.major_axis = panel.major_axis.tz_localize(pytz.utc)
    #print panel[stocks[0]].head(5)

    #close the connection
    l.Close()

    # Adjust data
    if adjusted:
        adj_cols = ['open', 'high', 'low', 'close']
        for ticker in panel.items:
            ratio = (panel[ticker]['price'] / panel[ticker]['close'])
            ratio_filtered = ratio.fillna(0).values
            for col in adj_cols:
                panel[ticker][col] *= ratio_filtered


    return [panel,start_time,end_time]
   ''' 


#定义一个函数，用于获取分钟数据，其中分钟数据也需要进行复权调整
def load_minute_data(indexes=None,stockList=None,start=None,end=None,adjusted=False,rolling_count= 10):

    """
    load stocks from Mongo
    """
    assert indexes is not None or stockList is not None
    """
    must specify stockList or indexes"""
    #对日期进行改造，提取的数据日期应该高于多于开始日期一个月，这样对于原数据有缓冲作用
    
    starts = start
    ends =end
    #start_time = pd.Timestamp(start,tz='UTC')
    #end_time = pd.Timestamp(end,tz='UTC')

    if start is None:
        start = "1990-01-01"

    if start is not None and end is not None:
        startdate = datetime.datetime.strptime(start, "%Y-%m-%d")
        enddate=datetime.datetime.strptime(end, "%Y-%m-%d")
        assert startdate < enddate, "start date is later than end date."

    data = OrderedDict()
    start = (datetime.datetime.strptime(start,'%Y-%m-%d')-datetime.timedelta(days=rolling_count+1)).strftime('%Y-%m-%d')
    end = (datetime.datetime.strptime(end,'%Y-%m-%d')+datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    l=LoadDataCVS(constants.IP,constants.PORT)
    l.Conn()

    if stockList=="hs300" or stockList=="zz500" or stockList=="sz50" or stockList=="all":
        stocks=l.getstocklist(stockList)
    else:
        stocks=stockList
    
    #print stocks

    if stocks is not None:
        for stock in stocks:
            stkd= l.getstockminute(stock,start,end)
            data[stock] = stkd
            #print data[stock].head(5)
            #print data[stock].tail(5)

    if indexes is not None:
        for name, ticker in iteritems(indexes):
            logger.info('Loading index: {} ({})'.format(name, ticker))
            stkd= l.getindexminute(indexes,start,end)
            data[name] = stkd

    #['open','high','low','close','volume','price','change',"code"]
    panel = pd.Panel(data)
    panel.minor_axis = ['open', 'high', 'low', 'close', 'volume', 'price','change','code']
    panel.major_axis = panel.major_axis.tz_localize(pytz.utc)
    #print panel[stocks[0]].head(5)

    #close the connection
    l.Close()

    # Adjust data
    if adjusted:
        adj_cols = ['open', 'high', 'low', 'close']
        for ticker in panel.items:
            ratio = (panel[ticker]['price'] / panel[ticker]['close'])
            ratio_filtered = ratio.fillna(0).values
            for col in adj_cols:
                panel[ticker][col] *= ratio_filtered

    starts = str(starts) + ' 09:25:00'
    start_time = pd.Timestamp(starts,tz='UTC')
    ends = str(ends) + ' 09:25:00'
    end_time = pd.Timestamp(ends,tz='UTC')
    return [panel,start_time,end_time]
    
    

def _load_raw_yahoo_data(indexes=None, stocks=None, start=None, end=None):
    """Load closing prices from yahoo finance.

    :Optional:
        indexes : dict (Default: {'SPX': '^GSPC'})
            Financial indexes to load.
        stocks : list (Default: ['AAPL', 'GE', 'IBM', 'MSFT',
                                 'XOM', 'AA', 'JNJ', 'PEP', 'KO'])
            Stock closing prices to load.
        start : datetime (Default: datetime(1993, 1, 1, 0, 0, 0, 0, pytz.utc))
            Retrieve prices from start date on.
        end : datetime (Default: datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc))
            Retrieve prices until end date.

    :Note:
        This is based on code presented in a talk by Wes McKinney:
        http://wesmckinney.com/files/20111017/notebook_output.pdf
    """
    assert indexes is not None or stocks is not None, """
must specify stocks or indexes"""

    if start is None:
        start = pd.datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)

    if start is not None and end is not None:
        assert start < end, "start date is later than end date."

    data = OrderedDict()
    if stocks is not None:
        for stock in stocks:
            logger.info('Loading stock: {}'.format(stock))
            stock_pathsafe = stock.replace(os.path.sep, '--')
            cache_filename = "{stock}-{start}-{end}.csv".format(
                stock=stock_pathsafe,
                start=start,
                end=end).replace(':', '-')
            cache_filepath = get_cache_filepath(cache_filename)
            if os.path.exists(cache_filepath):
                stkd = pd.DataFrame.from_csv(cache_filepath)
            else:
                stkd = DataReader(stock, 'yahoo', start, end).sort_index()
                stkd.to_csv(cache_filepath)
            data[stock] = stkd

    if indexes is not None:
        for name, ticker in iteritems(indexes):
            logger.info('Loading index: {} ({})'.format(name, ticker))
            stkd = DataReader(ticker, 'yahoo', start, end).sort_index()
            data[name] = stkd

    return data


def load_from_yahoo(indexes=None,
                    stocks=None,
                    start=None,
                    end=None,
                    adjusted=True):
    """
    Loads price data from Yahoo into a dataframe for each of the indicated
    assets.  By default, 'price' is taken from Yahoo's 'Adjusted Close',
    which removes the impact of splits and dividends. If the argument
    'adjusted' is False, then the non-adjusted 'close' field is used instead.

    :param indexes: Financial indexes to load.
    :type indexes: dict
    :param stocks: Stock closing prices to load.
    :type stocks: list
    :param start: Retrieve prices from start date on.
    :type start: datetime
    :param end: Retrieve prices until end date.
    :type end: datetime
    :param adjusted: Adjust the price for splits and dividends.
    :type adjusted: bool

    """
    data = _load_raw_yahoo_data(indexes, stocks, start, end)
    if adjusted:
        close_key = 'Adj Close'
    else:
        close_key = 'Close'
    df = pd.DataFrame({key: d[close_key] for key, d in iteritems(data)})
    df.index = df.index.tz_localize(pytz.utc)
    return df


def load_bars_from_yahoo(indexes=None,
                         stocks=None,
                         start=None,
                         end=None,
                         adjusted=True):
    """
    Loads data from Yahoo into a panel with the following
    column names for each indicated security:

        - open
        - high
        - low
        - close
        - volume
        - price

    Note that 'price' is Yahoo's 'Adjusted Close', which removes the
    impact of splits and dividends. If the argument 'adjusted' is True, then
    the open, high, low, and close values are adjusted as well.

    :param indexes: Financial indexes to load.
    :type indexes: dict
    :param stocks: Stock closing prices to load.
    :type stocks: list
    :param start: Retrieve prices from start date on.
    :type start: datetime
    :param end: Retrieve prices until end date.
    :type end: datetime
    :param adjusted: Adjust open/high/low/close for splits and dividends.
        The 'price' field is always adjusted.
    :type adjusted: bool

    """
    data = _load_raw_yahoo_data(indexes, stocks, start, end)
    panel = pd.Panel(data)
    # Rename columns
    panel.minor_axis = ['open', 'high', 'low', 'close', 'volume', 'price']
    panel.major_axis = panel.major_axis.tz_localize(pytz.utc)
    # Adjust data
    if adjusted:
        adj_cols = ['open', 'high', 'low', 'close']
        for ticker in panel.items:
            ratio = (panel[ticker]['price'] / panel[ticker]['close'])
            ratio_filtered = ratio.fillna(0).values
            for col in adj_cols:
                panel[ticker][col] *= ratio_filtered
    return panel


def load_prices_from_csv(filepath, identifier_col, tz='UTC'):
    data = pd.read_csv(filepath, index_col=identifier_col)
    data.index = pd.DatetimeIndex(data.index, tz=tz)
    data.sort_index(inplace=True)
    return data


def load_prices_from_csv_folder(folderpath, identifier_col, tz='UTC'):
    data = None
    for file in os.listdir(folderpath):
        if '.csv' not in file:
            continue
        raw = load_prices_from_csv(os.path.join(folderpath, file),
                                   identifier_col, tz)
        if data is None:
            data = raw
        else:
            data = pd.concat([data, raw], axis=1)
    return data
