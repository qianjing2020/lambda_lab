### Design a composite quality index to determine time series quality for forecasting
import numpy as np
import pandas as pd

from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from db_connect import dbConnect

db_c = dbConnect()

user = 'postgres'
password = 'yoaprenderia'
host = 'sauti-labs24.cfqelszozxua.us-east-1.rds.amazonaws.com'
port = '5432'
database = 'sautidb'

db_URI ='postgresql://'+user+':'+password+'@'+host+'/'+database
print(db_URI)

engine = create_engine(db_URI)

# fetch data from DB 
data = pd.read_sql('SELECT * FROM qc_wholesale', con=engine)
tablename = 'qc_wholesale'


# %%
# prescreening: remove valid data rows but containing NaN in mode_D and duplicates
df = data.copy()
df = df[~(df['duplicates'].isnull() | df['mode_D'].isnull())]


# %%
len(df)


# %%
# pick the ML candidate, seems there are really not great candidates 
df.sort_values(by='data_length', ascending=False).head()


# %%
# add an other col to get valid datapoint
df['data_points']=round(df['data_length']*df['completeness']-df['duplicates'], 0)
df.head()


# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

#encoder = OneHotEncoder()
#scaler = MinMaxScaler()

cat_cols = ['market', 'product', 'source']
num_cols = ['timeliness', 'data_length', 'completeness', 'duplicates', 'mode_D','data_points']

cat_vars = df[cat_cols]
num_vars = df[num_cols]

# log_scale_transformer = make_pipeline(
#     FunctionTransformer(func=np.log),
#     StandardScaler()
# )

column_trans = ColumnTransformer(
    [#('onehot_categorical', OneHotEncoder(), cat_cols), 
     ('scaled_numeric', MinMaxScaler(), num_cols), 
    ],
    remainder="drop",
)
X = column_trans.fit_transform(num_vars)


# %%
# tdf: transformed df 
tdf = df.copy()
tdf[num_cols]=X
tdf.head()


# %%
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
df.sort_values(by='DQI', ascending=False).head(99)


# %%
df['DQI_cat']=pd.qcut(df['DQI'], [0, .25, .5, .75, 0.9, 1.], labels = [ "poor", "medium", "fair", "good", "great"])
df.head()


# %%
plt.subplots(figsize=(10, 7))
sns.scatterplot('data_points', 'timeliness', hue='DQI_cat', data=df)
plt.show()


# %%
candidate = df[(df['DQI_cat']=='great') ]
print(len(candidate))
candidate.head(99)


# %%
# AWS DB opotion (this is Jesus's DB shared in the team)
import psycopg2

user = 'postgres'
password = 'yoaprenderia'
host = 'sauti-labs24.cfqelszozxua.us-east-1.rds.amazonaws.com'
port = '5432'
database = 'sautidb'
# con = psycopg2.connect(user=user, 
#                               host=host, 
#                               port=port, ctabase, 
#                               password=password)
# con


# %%
import time

start_time = time.time()

from sqlalchemy import create_engine
#+psycopg2
db_URI ='postgresql://'+user+':'+password+'@'+host+'/'+database
print(db_URI)
engine = create_engine(db_URI)


# upload dataframe to DB table
df.to_sql(tablename,
          con=engine, 
          if_exists='replace',
          index=False,
          chunksize=50,
#          dytype={"id":Integer,
#                 "source":object}
         )
elapsed_time = time.time()-start_time
print(f"--- QC table for {tablename} up in the cloud DB successfully in {elapsed_time/60} minutes! ---" )


