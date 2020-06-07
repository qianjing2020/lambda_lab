import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from db_connect import dbConnect
from data_preprocess import DataCleaning, DataQualityCheck

## define sale_type
sale_type = 'retail'
## Read data
db_c = dbConnect()
raw_data = db_c.read_stakeholder_db()

## Clean data
dc = DataCleaning()
df = raw_data.copy()
df  = dc.read_data(df)
df = dc.remove_zeros(df)
df = dc.convert_dtypes(df)

## load qc table to find candidate time series
tablename = 'qc_'+sale_type
qc = db_c.read_analytical_db(tablename)
candidate = qc[qc['DQI_cat']=='great'].iloc[1,:]
# Specify condition for selection of subset
# Note: cutoff-date to avoid large data gap, this can be automated later
cutoff_date = '2011-09-20'

# selected time series
MARKET = candidate['market']
PRODUCT = candidate['product']
SOURCE = candidate['source']
print(f'Market: {MARKET}, Product:{PRODUCT}, Source: {SOURCE}')

# apply filters
cond1 = (df['product']==PRODUCT)
cond2 = (df['source']==SOURCE)
cond3 = (df['market'] == MARKET)
# subset is the selected dataframe with both wholesale and retail and other infos    
subset = df[cond1 & cond2 & cond3].sort_values(by='date', ascending=True).set_index('date')

# get speficied sale type data
sale_cols = ['retail', 'wholesale']
sale_cols.remove(sale_type)
sale_df = subset.drop(columns=sale_cols)

# remove outliers before remove duplicates
y = sale_df[sale_type]
lower_bound, upper_bound = y.quantile(.05), y.quantile(.90)
idx = y.between(lower_bound, upper_bound)
sale_df = sale_df[idx]

# remove duplicates
idx = ~sale_df.index.duplicated(keep='first')
sale_df = sale_df[idx]

# construct data sample with complete time frame
dqc = DataQualityCheck()
yt = dqc.day_by_day(sale_df[sale_type]) 

# interpolate for missing values
yi = yt.interpolate(method='nearest')
plt.plot(yi)


# Check data stationarity
# plot autocorrelation function to decide lagging
# fig = plt.figure(figsize=(11, 10))
# ax = fig.add_subplot(211)
# plot_acf(yi, lags=50, ax=ax)
# ax2 = fig.add_subplot(212)
# plot_pacf(yi, lags=50, method='ols',ax=ax2)
# plt.show()

## transform data: boxcox, deseasonalize, detrend
# boxcox to achieve stationarity in variance
y_trans, lam = boxcox(yi.values.flatten())

y_trans = pd.Series(y_trans, index=yi.index)

results = STL(y_trans).fit()
results.plot()
plt.show()
# deseasonal, detrend:
y_dd = results.resid

###Predict using Holt Winter’s Exponential Smoothing (HWES), time series with trend and seasonal component
# a manual split

n_test = int(0.2 * len(y_trans))

train, test=y_trans[:-n_test], y_dd[-n_test:]

model = ExponentialSmoothing(train, 
                             trend='add', 
                             seasonal='add', 
                             seasonal_periods=30, 
                             damped=True)

hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
# optimized include smoothing level, slope, seasonal, and damping slope

pred = hw_model.predict(start=test.index[0], end=test.index[-1])

# inverse of boxcox: inv_boxcox(y_trans, lam)
pred = inv_boxcox(pred, lam)

fig = plt.figure(figsize=(20,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best')
plt.show()
