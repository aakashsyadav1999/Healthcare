import os 
import sys
from src.Healthcare.logger import logging
from src.Healthcare.exception import CustomException


from src.Healthcare.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.Healthcare.components.data_transformation import DataTransformation,DataTransformationConfig
from src.Healthcare.components.model_trainer import ModelTrainer,ModelTrainerConfig



if __name__=='__main__':
    logging.info("The execution has started")

    try:
        #These line of code will excute data ingestion file:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_config()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        logging.info("Data transformation has completed")

        logging.info("Model training has started")
        model_trainer=ModelTrainer()
        best_model,accuracy, prec, recall = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model training has completed!! ")


    except Exception as e:
        raise CustomException(e,sys)