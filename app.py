import os 
import sys
from src.Healthcare.logger import logging
from src.Healthcare.exception import CustomException


from src.Healthcare.components.data_ingestion import DataIngestion,DataIngestionConfig



if __name__=='__main__':
    logging.info("The execution has started")

    try:
        #These line of code will excute data ingestion file:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_config()





    except Exception as e:
        raise CustomException(e,sys)