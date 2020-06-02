import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime, date
pd.set_option('display.max_columns', 20)

class DataCleaning:
    """ method to clean data, apply to the whole data set (mixed time series)"""
    def __init__(self):
        pass
        
    def read_data(self, data=None):
        if data is None:
            print("Warning: No data provided")
        if (isinstance(data, pd.DataFrame) == False):
            print( "Input should be a dataframe!")
        else:
            df = pd.DataFrame(data)
            print('Data is fed to class object.')
        return df
              
    def remove_zeros(self, data):
        """clean all invalid entries
        cost cannot be 0, replace zeros with NaN"""
        df = data.copy()
        cols = ['wholesale', 'retail']
        
        df[cols] = df[cols].replace({0: np.nan})
        if np.prod(df['wholesale'] != 0):
            print('All zero values has been replaced with NaN successfully')
        else:
            print('Zero to NaN process not complete.')
        return df    
          
        # # remove str in wholesale retail columns
        # str_to_remove_list = ['Wholesale', 'retail','NaN']
        # df[df['wholesale'].isin(str_to_remove_list)] = np.NaN
        # df[df['retail'].isin(str_to_remove_list)] = np.NaN       

    def convert_dtypes(self, data):
        """change each column to desired data type"""
        df = data.copy()
        # # change date to datetime
        # df['date'] = pd.to_datetime(df['date'])

        # change num dtype to float
        df['wholesale'] = df['wholesale'].astype('float')
        df['retail'] = df['retail'].astype('float')
      
        # change text col to categorical
        str_cols = ['market', 'product', 'country', 'currency']
        for item in str_cols:
            df[item] = df[item].astype('category')
        
        print('Data type converted. Numericals converted to float, date to datatime type, and non-numericals to category.')
       
        return df
     

class DataQualityCheck:
    """contain methods for quality check for one time series"""
    def __init__(self):
        pass
    
    def read_data(self, data=None):
        if data is None:
            print("Warning: No data provided")
        if (isinstance(data, pd.Series) == False) & (isinstance(data.index, pd.DatetimeIndex)):
            print("Data needs to be pandas series with datetime index!")
        else:
            df = pd.Series(data)
        return df
        
    def remove_duplicates(self, df):
        """remove duplicated rows, keep the first"""
        y = df.copy()
        rows_rm = y.index.duplicated(keep='first')
        if np.sum(rows_rm):
            y = y[~rows_rm]
        return y
        
    def remove_outliers(self, df):
        """remove outliers from a series"""        
        y = df.copy()
        lower_bound, upper_bound = y.quantile(.05), y.quantile(.95)
        y = y[y.iloc[:,0].between(lower_bound[0], upper_bound[0])]
        return y

    def day_by_day(self, df):
        """construct time frame and create augumented time series"""
        y = df.copy()
        START, END = y.index.min(), y.index.max()        
        # construct a time frame from start to end
        date_range = pd.date_range(start=START, end=END, freq='D')
        time_df = pd.DataFrame([], index=date_range)
        # this is time series framed in the complete day-by-day timeframe
        y_t = time_df.merge(y, how='left', left_index=True, right_index=True)
        return y_t

    def generate_QC(self, df, figure_output=0):
        """ 
        Input:  y: time series with sorted time index

        Output: time series data quality metrics
            start: start of time series
            end: end of time seires
            timeliness: gap between the end of time seires and today, days. 
                    0 means sampling is up to today, 30 means the most recent data was sampled 30 days ago.
            data_length: length of available data in terms of days
            completeness: not NaN/total data in a complete day-by-day time frame,
                    0 means all data are not valid, 1 means data is completed on 
            duplicates: number of data sampled on same date, 0: no duplicates, 10: 10 data were sampled on a same date
            mode_D: the most frequent sampling interval in time series, days, 
                this is important for determing forecast resolution

        """
        y = df.copy()
        # construct time frame and create augumented time series
        START, END = y.index.min(), y.index.max()
        TIMELINESS = (datetime.now()-END).days
        
        # this is time series framed in the complete day-by-day timeframe
        y_t = day_by_day(y) 

        # completeness
        L = len(y_t)
        L_nan = y_t.isnull().sum()
        COMPLETENESS = (1-L_nan/L)[0]
        COMPLETENESS = round(COMPLETENESS, 3)
        DATA_LEN = L

        if COMPLETENESS == 0 | DATA_LEN == 1:
            # no data or 1 datum
            DUPLICATES = np.nan
            MODE_D = np.nan

        else:
            # some data exist

            timediff = pd.DataFrame(np.diff(y.index.values), columns=['D'])
            x = timediff['D'].value_counts()
            x.index = x.index.astype(str)
            # x is value counts of differences between all adjecent sampling dates for one time series

            if x.empty:
                # only one data available, keep row for future data addition
                DUPLICATES = 0
                MODE_D = 0

            elif any(x.index == '0 days') | len(x) == 1:
                # duplicates exists, and all data occur on the same date
                DUPLICATES = x[0]
                MODE_D = 0

            elif any(x.index == '0 days') | len(x) > 1:
                # duplicates exists and data not equally spaced
                DUPLICATES = x[0]
                MODE_D = x[~(x.index == '0 days')].index[0]

            else:  # elif ('0 days' not in x.index):
                # no duplication
                DUPLICATES = 0
                MODE_D = x.index[0]

        START = str(START.date())
        END = str(END.date())
        QC_i = [START, END, TIMELINESS,
                DATA_LEN, COMPLETENESS, DUPLICATES, MODE_D]

        if figure_output == 1:
            # a small plot indicating sampling scheme
            ax = sns.heatmap(y_t.isnull(), cbar=False)
            plt.show()

        return QC_i
