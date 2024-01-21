import os
import sys
from src.ML_Project_1.exception import CustomException
from src.ML_Project_1.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')

def read_sql_data():
    logging.info('Reading SQL database strarted')
    try:
        mydb=pymysql.connect(           # Connect with SQL Database
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established",mydb)
        df=pd.read_sql_query('Select * from feedback',mydb)
        print(df.head())

        return df
    
    except Exception as ex:
        raise CustomException(ex)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e.sys)