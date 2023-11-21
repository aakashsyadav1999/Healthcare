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


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

        

    def initiate_data_config(self):

        try:
            df = pd.read_csv(r'D:\vscode\Healthcare\Notebook\data\healthcare_dataset.csv')

            logging.info("Reading CSV file")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),
                        exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,
                      index = False,
                      header = True)
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header = True)

            logging.info("Data ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        

        except Exception as e:
            raise CustomException(e,sys)