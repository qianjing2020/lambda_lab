import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime

from TimeBasedCV import TimeBasedCV
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
candidate = qc[qc['DQI_cat']=='great'].iloc[0,:]
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
lower_bound, upper_bound = y.quantile(.05), y.quantile(.95)
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

# Check data stationarity
# plot autocorrelation function to decide lagging
fig = plt.figure(figsize=(11, 10))
ax = fig.add_subplot(211)
plot_acf(yi, lags=50, ax=ax)
ax2 = fig.add_subplot(212)
plot_pacf(yi, lags=50, method='ols',ax=ax2)
plt.show()

#  Predict using Holt Winter’s Exponential Smoothing (HWES), time series with trend and seasonal component
# a manual split
T = int(0.8*len(yi))
train, test = yti.iloc[:T, 0], yi.iloc[T:, 0]

model = ExponentialSmoothing(train,
                             trend='add',
                             seasonal='add',
                             seasonal_periods=365,
                             damped=True)

hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)


tscv = TimeBasedCV(
    train_period=30,
    test_period=7,
    freq='days')

tscv.split(y_i, validation_split_date=datetime.date(2019, 2, 1), date_column=y_i.index)

# get number of splits
#tscv.get_n_splits()

# computer average test sets score
