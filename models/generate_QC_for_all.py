import sys
sys.path.insert(0, '/Users/jing/Documents/LambdaSchool/lambda_lab/')
sys.path.insert(0, '/Users/jing/Documents/LambdaSchool/lambda_lab/models')

import pdb
from settings import *

from sqlalchemy import create_engine
import psycopg2
import pymysql
import pandas as pd
import numpy as np


from db_connect import dbConnect
from data_preprocess import DataCleaning, DataQualityCheck

# obtain data
db_c = dbConnect()
raw_data = db_c.read_stakeholder_db()
print(raw_data.head())

df0 = raw_data.copy()
# clean data
df = DataCleaning(raw_data)
df = df.simplify_header()
df = DataCleaning.clean_entry(df)
df = df.convert_dtypes()

print(df.head())

pdb.set_trace()
# quality check
qc = DataQualityCheck(df)
qc


sale_type = 'retail'
start_time = time.time()
# create a summarize table for data quality
col_names = ['market', 'product', 'source', 'start', 'end', 'timeliness', 'data_length', 'completeness', 'duplicates', 'mode_D']

m = len(MARKET_LIST)*len(PRODUCT_LIST)*len(SOURCE_LIST)
n = len(col_names)
anticipated_QC_size = m*n

QC = [[] for _ in range(m)] 

i = 0
for MARKET in MARKET_LIST:
    for PRODUCT in PRODUCT_LIST:
        for SOURCE in SOURCE_LIST:
            
            # apply filters
            cond1 = (df['product_agg']==PRODUCT)
            cond2 = (df['source']==SOURCE)
            cond3 = (df['market']==MARKET)
                        
            subset = df[cond1 & cond2 & cond3].sort_values(by='date', ascending=True).set_index('date')
            
            # this is the sale time series
            sale = subset[[sale_type]] 
            
            if sale.empty:
                break
            
            if len(sale)==sale.isnull().sum().values[0]:
                break
                
            else:
                QC_i = generate_QC(sale)
                QC[i] = QC_i
                i = i+1

# write to DQ dataframe
QC_df = pd.DataFrame(columns=col_names, data = QC)

elapsed_time = time.time()-start_time
print(f"--- QC table generated for {sale_type} successfually in {elapsed_time/60} minutes! ---" )
