import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from helpers1 import plot_AFC_PAFC, test_stationarity


sns.set_style("whitegrid")
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)


if __name__ == '__main__':
    ts = pd.read_csv('data1/my_time_series.csv', index_col='time', usecols=['time', 'X'])
    ts_log = np.log(ts)
    mywindow = 40
    test_stationarity(ts_log, mywindow=mywindow, filename='ts_log')

    # Making stationary as seen in exercise session 9
    moving_avg = ts_log.rolling(window=mywindow).mean()
    plt.plot(ts_log)
    plt.plot(moving_avg, color='red')
    ts_log_moving_avg_diff = ts_log - moving_avg
    ts_log_moving_avg_diff.dropna(inplace=True)
    test_stationarity(ts_log_moving_avg_diff, mywindow=mywindow, filename='ts_log_moving_avg_diff')


    # Determine p and q values
    partial = True
    lags = 20
    lag_acf = acf(ts_log_moving_avg_diff, nlags=200, fft=False)
    lag_pacf = pacf(ts_log_moving_avg_diff, nlags=200, method='ols')
    plot_AFC_PAFC(ts_log_moving_avg_diff, lag_acf, lag_pacf)
    # Do it with the builtin function (looks nicer)
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=(12, 4))
    ax2 = fig.add_subplot(111)
    plt.xticks(range(0, lags+1, 1))
    if partial:
        fig = sm.graphics.tsa.plot_pacf(ts_log_moving_avg_diff, lags=lags, ax=ax2, title='')
        plt.ylabel('PACF')
    else:
        fig = sm.graphics.tsa.plot_acf(ts_log_moving_avg_diff, lags=lags, ax=ax2, title='')
        plt.ylabel('ACF')
    plt.xlabel('lag')

    plt.tight_layout()
    if partial:
        plt.savefig('../report/images/partialautocorrelation.png')
    else:
        plt.savefig('../report/images/autocorrelation.png')
