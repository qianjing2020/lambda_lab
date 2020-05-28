import sys
sys.path.insert(0, '/Users/jing/Documents/LambdaSchool/lambda_lab/')
import numpy as np
import pandas as pd

import pymysql
import psycopg2
from sqlalchemy import create_engine

from settings import *

class dbConnect:

    def __init__(self, name='wholesale'):
        self.name = name
        self.df = [] # create an empy dataframe
    
    def read_stakeholder_db(self):
        db_URI = 'mysql+pymysql://' + stakeholder_db_user + ':' + \
            stakeholder_db_password + '@' + stakeholder_db_host + '/' + stakeholder_db_name
        engine = create_engine(db_URI)    

        data = pd.read_sql(
            "SELECT * FROM platform_market_prices2", con=engine)        
        return data
    
    

    def initiate_analytical_db(self, df):
        con = psycopg2.connect(
            user=aws_db_user,
            host=aws_db_host,
            port=aws_db_port,
            database=aws_db_name,
            password=aws_db_password)
        db_URI = 'postgresql://' + aws_db_user + ':' + aws_db_password + '@' + aws_db_host + '/' + aws_db_name
        engine = create_engine(db_URI)
        tablename = self.name.lower()

        df.to_sql(tablename, con=engine, if_exist='replace', index=False, chuncksize=100)
        con.commit()
        con.close()

    def migrate_analyticalDB(self):
        pass #raw = read_stakeholderDB()


