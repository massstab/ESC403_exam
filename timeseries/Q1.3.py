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

ts = pd.read_csv('data1/HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv')
ts['Time'] = pd.to_datetime(ts['Time'])
ts.set_index('Time', inplace=True)
ts.rename(columns={"Anomaly (deg C)": "anomaly"}, inplace=True)
ts.drop(columns=['Lower confidence limit (2.5%)', 'Upper confidence limit (97.5%)'], inplace=True)


one = False
two = False
three = True
three_experimental = False

if one:
    # Visualize and test stationarity
    test_stationarity(ts, mywindow=12, filename='ts_moving_avg_B')


    # differencing and visualize again
    ts_diff = ts - ts.shift()
    ts_diff.dropna(inplace=True)
    test_stationarity(ts_diff, mywindow=12, filename='ts_moving_avg_diff_B')

    # Determine p and q values
    partial = True
    lags = 24
    fig = plt.figure(figsize=(12, 4))
    ax2 = fig.add_subplot(111)
    plt.xticks(range(0, lags+1, 2))
    if partial:
        fig = sm.graphics.tsa.plot_pacf(ts_diff, lags=lags, ax=ax2, title='', zero=False)
        plt.ylabel('PACF')
    else:
        fig = sm.graphics.tsa.plot_acf(ts_diff, lags=lags, ax=ax2, title='', zero=False)
        plt.ylabel('ACF')
    plt.xlabel('lag')

    plt.tight_layout()
    if partial:
        plt.savefig('../report/images/partialautocorrelation_B.png')
    else:
        plt.savefig('../report/images/autocorrelation_B.png')
    plt.show()
if two:
    # ARIMA model
    model = ARIMA(ts, order=(12, 0, 2))
    results_ARIMA = model.fit(disp=-1)
    plt.clf()
    plt.plot(ts)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts['anomaly'])**2))
    print(results_ARIMA.summary())


    # Plot residual errors
    plt.clf()
    residuals = pd.DataFrame(results_ARIMA.resid)
    # fig, ax = plt.subplots(1,2, figsize=(14, 6))
    # residuals.plot(title="Residuals", ax=ax[0], legend=False)
    # residuals.plot(kind='kde', title='Density', ax=ax[1], legend=False)
    # plt.tight_layout()
    # plt.savefig('../report/images/res_dens.png')


    # Prediction for the average temperature anomaly from 2010-01 - 2021-03
    plt.clf()
    plt.cla()
    # fig, ax = plt.subplots(figsize=(10, 5))
    # results_ARIMA.plot_predict(dynamic=False, start='2010-01', end='2021-03', ax=ax)
    # plt.tight_layout()
    # plt.savefig('../report/images/forecast_2010-2021.png')
    # plt.show()

    predictions_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.plot(ts, c='r')
    plt.plot(predictions_ARIMA)
    print('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts['anomaly'])**2)/len(ts)))
    plt.tight_layout()
    plt.savefig('../report/images/forecast_2010-2021_RMSE.png')
    # plt.show()

if three:
    # ARIMA model
    ts_diff = ts - ts.shift()
    ts_diff.dropna(inplace=True)
    plt.clf()
    plt.cla()
    model = ARIMA(ts_diff, order=(12, 0, 2))
    results_ARIMA = model.fit()
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.tight_layout()
    results_ARIMA.plot_predict(dynamic=False, start='2021-03', end='2050-01', ax=ax)
    plt.savefig(f'tmp/forecast_2021-2050.png')

if three_experimental:
    # ARIMA model
    ts_diff = ts - ts.shift()
    ts_diff.dropna(inplace=True)
    values_k = [11, 12, 13]
    values_i = [0, 1, 2]
    for i in values_i:
        for k in values_k:
            plt.clf()
            plt.cla()
            model = ARIMA(ts_diff, order=(i, 0, k))
            results_ARIMA = model.fit()
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.tight_layout()
            results_ARIMA.plot_predict(dynamic=False, start='2021-03', end='2050-01', ax=ax)
            plt.title(f'{i} 0 {k}')
            plt.savefig(f'tmp/{i} 0 {k}.png')