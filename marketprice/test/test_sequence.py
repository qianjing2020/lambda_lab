import context
from modules.data_preprocess import DataCleaning
from modules.db_connect import dbConnect

# test time series
MARKET = 'Masindi'
PRODUCT = 'Cowpeas'
SOURCE = 'EAGC-RATIN'

db_c = dbConnect()
raw_data = db_c.read_stakeholder_db()
## Clean data
dc = DataCleaning()
df = raw_data.copy()
df  = dc.read_data(df)
df = dc.remove_zeros(df)
df = dc.convert_dtypes(df)
print('Data cleanning modules work!')

# create subset for testing DataQualityCheck module
cond1 = (df['product']==PRODUCT)
cond2 = (df['source']==SOURCE)
cond3 = (df['market']==MARKET)

subset = df[cond1 & cond2 & cond3].sort_values(by='date', ascending=True).set_index('date')
sale_type = 'retail'
# this is the sale time series
sale = subset[[sale_type]] 