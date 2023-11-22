import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from sklearn.model_selection import train_test_split

from src.Healthcare.exception import CustomException
from src.Healthcare.logger import logging

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train_data.csv")
    test_data_path:str = os.path.join("artifacts","test_data.csv")
    raw_data_path:str = os.path.join("artifacts","raw_data.csv") 

#Data Ingestion class
class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

        
    #initiate data class
    def initiate_data_config(self):

        try:
            #Reading data
            df = pd.read_csv(r'D:\vscode\Healthcare\Notebook\data\healthcare_dataset.csv')
            
            #Convert target column from categorical to numerical column
            df['Test Results'] = df['Test Results'].replace({'Normal':0,"Inconclusive":1,"Abnormal":2})

            df = df.drop(columns=['Date of Admission','Billing Amount','Discharge Date','Name'],axis=1)

            df.columns = df.columns.str.replace(" ","_")
                     
            #Logging message
            logging.info("Reading CSV file")

            #Self ingestion training and test data path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),
                        exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,
                      index = False,
                      header = True)
            
            #splitting data into train and test part
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header = True)

            logging.info("Data ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        #Custom-Exception
        except Exception as e:
            raise CustomException(e,sys)