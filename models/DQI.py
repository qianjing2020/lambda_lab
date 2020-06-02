"""Design a composite quality index to determine time series quality for forecasting"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from db_connect import dbConnect

# get qc table
db_c = dbConnect()
tablename = 'qc_retail'

df = db_c.read_analytical_db(tablename=tablename)

# add an other col to get valid datapoint
df['data_points']=round(df['data_length']*df['completeness']-df['duplicates'], 0)

# convert mode_D to float
df['mode_D'] = df['mode_D'].astype(str).str.rstrip(' days').astype(float)

# transfer num columns to minmax scale
cat_cols = ['market', 'product', 'source']
num_cols = ['timeliness', 'data_length', 'completeness', 'duplicates', 'mode_D','data_points']

cat_vars = df[cat_cols]
num_vars = df[num_cols]

column_trans = ColumnTransformer(
    [('scaled_numeric', MinMaxScaler(), num_cols), 
    ],
    remainder="drop",
)

X = column_trans.fit_transform(num_vars)

# tdf: transformed df 
tdf = df.copy()
tdf[num_cols]=X

# Rank the data for ML candidate using data quality index (DQI)
# Higher DQI, better data quality
# DQI based on six data quality dimensions:
D1, W1 = tdf['data_length'], 0.6
D2, W2 = tdf['completeness'], 0.3
D3, W3 = 1-tdf['mode_D'], 0.9
D4, W4 = 1-tdf['timeliness'], 0.9
D5, W5 = 1-tdf['duplicates'], 0.3
D6, W6 = tdf['data_points'], 0.9

df['DQI'] = D1*W1 + D2*W2 + D3*W3 + D4*W4 + D5*W5 + D6*W6

df['DQI_cat']=pd.qcut(df['DQI'], [0, .25, .5, .75, 0.9, 1.], labels = [ "poor", "medium", "fair", "good", "great"])

plt.subplots(figsize=(10, 7))
sns.scatterplot('data_points', 'timeliness', hue='DQI_cat', data=df)
plt.show()

candidate = df[(df['DQI_cat']=='great') ]
print(f'len(candidate) time series fall into "great" DQI category' )
print(candidate)

# upload dataframe to DB table
db_c.populate_analytical_db(df, tablename=tablename)


