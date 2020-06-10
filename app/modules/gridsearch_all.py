"""grid-search for all database series using holter-winter method"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocess import DataCleaning, DataQualityCheck
from db_connect import dbConnect
from grid_search import grid_search
from forecast_models import ModelConfig
import time 
## test time series
sale_type = 'retail'

## get quality info for screening time series
db_c = dbConnect()
qc_tablename = 'qc_'+sale_type
qc_table = db_c.read_analytical_db(qc_tablename)
# return only great data quality
qc = qc_table[qc_table['DQI_cat'] == 'great']

## get raw data, clean data
db_c = dbConnect()
raw_data = db_c.read_stakeholder_db()
dc = DataCleaning()
df = raw_data.copy()
df  = dc.read_data(df)
df = dc.remove_zeros(df)
df = dc.convert_dtypes(df)
# print('Data cleaned!')

# create subset and loop through 
product_list = qc['product'].unique()
market_list = qc['market'].unique()
source_list = qc['source'].unique()

# grid search result init
results = []
mc = ModelConfig()

qc_id_list = qc['qc_id'].values

start_time = time.time()
for QC_ID in qc_id_list:
    # loop through candidate time series for grid search
    PRODUCT = qc[qc['qc_id']==QC_ID]['product'].values[0]
    MARKET = qc[qc['qc_id']==QC_ID]['market'].values[0]
    SOURCE = qc[qc['qc_id']==QC_ID]['source'].values[0]
    
    cond1 = (df['product'] == PRODUCT)
    cond2 = (df['source'] == SOURCE)
    cond3 = (df['market'] == MARKET)
    subset = df[cond1 & cond2 & cond3].sort_values(by='date', ascending=True).set_index('date')
    
    # this is the sale time series
    sale = subset[[sale_type]]

    # time series clean up
    dqc = DataQualityCheck()
    sale = dqc.remove_outliers(sale, 0.05, 0.8)
    sale = dqc.remove_duplicates(sale)
    
    # grid search
    data = sale.to_numpy().flatten()

    n_test = 7 # number of observation used for test 
    max_length = len(data) - n_test  # max length used as history

    # config list for one time series
    cfg_list = mc.exp_smoothing_configs(seasonal=[0, 12])

    #print(f'{len(cfg_list)} configurations: {cfg_list}')

    # grid search
    scores = grid_search(data, cfg_list, n_test)

    # list top 1 configs
    # for cfg, error in scores[:10]:
    #     print(cfg, error)
    # top score
    cfg_selected, error = scores[1]
    result = [QC_ID, cfg_selected, error]
    results.append(result)

elapsed_time = time.time() - start_time

print(f'Elapsed time is {elapsed_time/60} min for grid searching {len(qc)} time seires.')

save_data = pd.DataFrame(results, columns=['qc_id', 'hw_params', 'RMSE'])
#'t', 'd', 's', 'p', 'b', 'r'])
## columns: qc_id, trend type, dampening type, seasonality type, seasonal period, Box-Cox transform, removal of the bias when fitting

# populate database with new model configuration table
tablename = 'hw_config_' + sale_type
db_c.populate_analytical_db(save_data, tablename)

# save to csv file
savecsv = 'hw_config_' + sale_type +'.csv'
save_data.to_csv(savecsv)

# pickle dataframe
savepickle =  'hw_config_' + sale_type +'_pickle'
save_data.to_pickle(savepickle)




