import context
from modules.data_preprocess import DataCleaning, DataQualityCheck
from modules.db_connect import dbConnect
import matplotlib.pyplot as plt 

# test time series
  
MARKET = 'Dar es salaam'
PRODUCT = 'Morogoro Rice'
SOURCE = 'EAGC-RATIN'
sale_type = 'retail'

db_c = dbConnect()
raw_data = db_c.read_stakeholder_db()
## Clean data
dc = DataCleaning()
df = raw_data.copy()
df  = dc.read_data(df)
df = dc.remove_zeros(df)
df = dc.convert_dtypes(df)
print('Data cleaned!')

# create subset for testing DataQualityCheck module
cond1 = (df['product']==PRODUCT)
cond2 = (df['source']==SOURCE)
cond3 = (df['market']==MARKET)

subset = df[cond1 & cond2 & cond3].sort_values(by='date', ascending=True).set_index('date')
# this is the sale time series
sale = subset[[sale_type]]

# time series clean up
dqc = DataQualityCheck()
sale = dqc.remove_outliers(sale, 0.05, 0.8)
sale = dqc.remove_duplicates(sale)

# plt.plot(sale,'.')
# plt.show()
