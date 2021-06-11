import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

sns.set_style("whitegrid")
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)


def plot_AFC_PAFC(ts, lag_acf, lag_pacf):
    from matplotlib.pyplot import figure
    figure(figsize=(12, 4))

    # Plot ACF:
    plt.subplot(111)
    plt.plot(lag_acf[1:])
    plt.xticks(range(1, 200, 10))
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.xlabel('lag')
    plt.ylabel('autocorrelation')
    # plt.title('Autocorrelation Function')

    # Plot PACF:
    # plt.subplot(122)
    # plt.plot(lag_pacf)
    # plt.xticks(range(0, 20, 1))
    # plt.axhline(y=0, linestyle='--', color='gray')
    # plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    # plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.savefig('../report/images/autocorrelation_identification.png')
    plt.show()

def test_stationarity(ts, mywindow, filename=None):
    # Plot rolling statistics:
    rolmean = ts.rolling(window=mywindow).mean()
    rolstd = ts.rolling(window=mywindow).std()
    figure(figsize=(14, 6), dpi=150)
    orig = plt.plot(ts, color='blue', label='Original', linewidth=0.6)
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Standard deviation')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('X')
    plt.tight_layout()
    # plt.title('Rolling Mean & Standard Deviation')

    # Perform the Dickey-Fuller test
    stats = ['Test Statistic', 'p-value', 'Lags', 'Observations']
    df_test = adfuller(ts, autolag='AIC')
    df_results = pd.Series(df_test[0:4], index=stats)
    for key, value in df_test[4].items():
        df_results['Critical Value (%s)' % key] = value
    print('\n--------------------------------------')
    print(f'results Dickey-Fuller for {filename}:\n', df_results)
    if filename:
        plt.savefig(f'../report/images/{filename}.png')
    df_results.to_csv(f'../report/data/dickey-fuller_{filename}.csv')
    # plt.show(block=False)

