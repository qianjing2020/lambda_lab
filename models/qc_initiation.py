import sys
sys.path.insert(0, '/Users/jing/Documents/LambdaSchool/lambda_lab/')
sys.path.insert(0, '/Users/jing/Documents/LambdaSchool/lambda_lab/models')

from settings import *
import time
from datetime import datetime, date
from sqlalchemy import create_engine
import psycopg2
import pymysql
import pandas as pd
import numpy as np

from db_connect import dbConnect
from data_preprocess import DataCleaning, DataQualityCheck

# User input: sale type 
sale_type = 'retail'

## read data from stakeholder's database
# instantiate db class
db_c = dbConnect()
raw_data = db_c.read_stakeholder_db()
print('data retrieved for local analysis')

## Clean data
dc = DataCleaning()
df = raw_data.copy()
df  = dc.read_data(df)
df = dc.remove_zeros(df)
df = dc.convert_dtypes(df)
print('data cleaned')

## Generate quality table with specified data quality dimensions
# instantiate qc class
qc = DataQualityCheck()


# Data quality dimension
col_names = ['market', 'product', 'source', 'start', 'end', 'timeliness', 'data_length', 'completeness', 'duplicates', 'mode_D']

# All product list
PRODUCT_LIST = df['product'].unique().tolist()
# All market list
MARKET_LIST = df['market'].unique().tolist()
# All source list
SOURCE_LIST = df['source'].unique().tolist()
m = len(MARKET_LIST)*len(PRODUCT_LIST)*len(SOURCE_LIST)
n = len(col_names)
print(f'Anticipate qc talbe size is {m*n} entries')

start_time = time.time()

# initialize QC table
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
                sale = qc.remove_duplicates(sale)
                sale = qc.remove_outliers(sale)
                QC_i = qc.generate_QC(sale, figure_output=0)
                QC[i] = [MARKET, PRODUCT, SOURCE] + QC_i
                i = i+1

# write to DQ dataframe
QC_df = pd.DataFrame(columns=col_names, data = QC)

# remove valid data rows but containing NaN in mode_D and duplicates
QC_df = QC_df[~(QC_df['duplicates'].isnull() | QC_df['mode_D'].isnull())]

# add id column
QC_df['qc_id'] = QC_df.index
cols = QC_df.columns.tolist()
# move id to first column location
cols = [cols[-1]] + cols[:-1]
QC_df = QC_df[cols]
# populate database with new qc table
tablename = 'qc_' + sale_type
db_c.populate_analytical_db(QC_df, tablename)

elapsed_time = time.time()-start_time
print(f"--- QC table generated for {sale_type}  and successfually initiated in AWS db. Elapsed time ={elapsed_time/60} minutes! ---" )
