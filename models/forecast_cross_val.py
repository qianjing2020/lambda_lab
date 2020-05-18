from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

import datetime
from TimeBasedCV import TimeBasedCV

df = pd.read_sql_table('data', 'sqlite:///../data/mydb.db')

# Specify condition for selection of subset
# Note: cutoff-date to avoid large data gap, this can be automated later
cutoff_date = '2011-09-20'
Market = 'Lira'
date_selected = (df['date'] > cutoff_date)
market_selected = (df['market'] == Market)
product_selected = (df['product'] == 'Maize')

condition = date_selected & market_selected & product_selected
df = df[condition]

retail = df[['date', 'retail']]
retail = retail.sort_values('date')
retail.set_index('date', inplace=True)

wholesale = df[['date', 'wholesale']].sort_values('date')
wholesale.set_index('date', inplace=True)
wholesale.loc['2011-09-01':'2011-09-30']

# construct complete time frame
date_range = pd.date_range(start=retail.index[0], end=retail.index[-1], freq='D')
time_df = pd.DataFrame([], index=date_range)

# construct data sample with complete time frame
yt = time_df.merge(wholesale, how='outer', left_index=True, right_index=True)
# interpolate for missing values
yt_n = yt.interpolate(method='nearest')
y_i = yt_n.reset_index()
'''
# Check data stationarity
# plot autocorrelation function to decide lagging
plot_acf(y_i, lags=500)
plt.show()

plot_pacf(y_i, lags=50, method='ols')
plt.show()
'''

#  Predict using Holt Winter’s Exponential Smoothing (HWES), time series with trend and seasonal component
# a simple split
T = int(0.8*len(y_i))
train, test = y_i.iloc[:T, 0], y_i.iloc[T:, 0]

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
