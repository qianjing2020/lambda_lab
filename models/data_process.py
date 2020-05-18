import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class data_cleaning:
    def __init__(self, description):
        self.description =
        

    def simpleHeader(df):
        df = df.rename(columns = lambda x: x.lower())
        cols = df.columns.tolist()
        df = df.rename(columns={cols[-3]: 'retail', cols[-2]: 'wholesale'})
        print(f'column header renamed {df.columns.tolist()}'')
        return df 

    def convertFormat(df):
        str_to_remove_list = ['Wholesale', 'retail','NaN']
        
        df[df['wholesale'].isin(str_to_remove_list)] = np.NaN
        df[df['retail'].isin(str_to_remove_list)] = np.NaN
        
        df['wholesale'] = df['wholesale'].astype('float')
        df['retail'] = df['retail'].astype('float')
        
        df['date'] = pd.to_datetime(df['date'])

        str_cols = ['market', 'product', 'country', 'currency']

        for item in str_cols:
            df[item]=df[item].astype('category') # which will by default set the length to the max len it encounters        
        

        # replace zeros with NaN
        cols = ['wholesale', 'retail']
        df[cols] = df[cols].replace({0: np.nan})
        if np.prod(df['wholesale'] != 0) == 1:
            print('All zero values has been replaced with NaN successfually')
        
        print('Success. Numericals converted to float, date to datatime type, and non-numericals to category, zero values to NaN.')

        return df
        
    def removeOutliers(df):
        def remove_outliers(x):
            lower_bound, upper_bound = x.quantile(.05), x.quantile(.95)
            x = x[x.between(lower_bound, upper_bound)]
            return x

        df['wholesale'] = remove_outliers(df['wholesale'])
        df['retail'] = remove_outliers(df['retail'])
        return df

    def removeDuplicatedRow(df):
        rows_rm = df.duplicated('date', keep='last')

        if np.sum(rows_rm):
        df = df[~rows_rm]
        

    def saveData(df):
        df.to_csv('../data/cleaned_data.csv', index=False)
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///../data/mydb.db', echo=False)
        df.to_sql('data', con=engine, if_exists='replace',
            index_label='id')
        print('Data has been backuped suscessfually to local drive as csv and db')



