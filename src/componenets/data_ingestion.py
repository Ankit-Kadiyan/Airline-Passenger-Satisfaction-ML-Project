import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import read_sql_data
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')   # Path to save train.csv file
    test_data_path:str=os.path.join('artifacts','test.csv')     # Path to save test.csv file
    raw_data_path:str=os.path.join('artifacts','raw.csv')       # Path to save raw.csv file

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ##reading code
            # df=read_sql_data()
            df=pd.read_csv(os.path.join('notebook\\data','raw_data.csv'))
            logging.info('Reading Completed MySQL Database')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)   # Creating artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)              # Save the raw.csv file
            
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)               # Splitting the Data into train and test Data
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)     # Save the train.csv File
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)       # Save the test.csv file

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)