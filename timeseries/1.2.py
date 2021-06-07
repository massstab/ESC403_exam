from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sns.set_style("whitegrid")
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)


def test_stationarity(ts):
    # Plot rolling statistics:
    rolmean = ts.rolling(window=12).mean()
    rolstd = ts.rolling(window=12).std()
    orig = plt.plot(ts, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')

    # Perform the Dickey-Fuller test
    stats = ['Test Statistic', 'p-value', 'Lags', 'Observations']
    df_test = adfuller(ts, autolag='AIC')
    df_results = pd.Series(df_test[0:4], index=stats)
    for key, value in df_test[4].items():
        df_results['Critical Value (%s)' % key] = value
    print(df_results)
    plt.savefig('../report/images/dickey-fuller-test.svg')
    plt.show(block=False)

ts = pd.read_csv('data1/my_time_series.csv', index_col='time', usecols=['time', 'X'])
ts_log = np.log(ts)
test_stationarity(ts_log)

# plt.plot(ts)
#
# fig, ax = plt.subplots(figsize=(16, 8))
# x = ts['time'][:]
# y = ts['X'][:]
# ax.semilogy(x, y)
# ax.set(xlabel='time', ylabel='X')
#

# plt.show()
