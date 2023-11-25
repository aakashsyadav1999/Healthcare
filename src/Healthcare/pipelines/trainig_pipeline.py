import os 
import sys
from src.Healthcare.logger import logging
from src.Healthcare.exception import CustomException


from src.Healthcare.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.Healthcare.components.data_transformation import DataTransformation,DataTransformationConfig
from src.Healthcare.components.model_trainer import ModelTrainer,ModelTrainerConfig


import mlflow
import dagshub
from urllib.parse import urlparse


class TrainingPipeline:
    def start_model_training(self):
        """
        start_model_training function will return the best model with score
        """

        try:
            logging.info("Model training has started!!")

            logging.info("Data ingestion has started!!")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path=data_ingestion.initiate_data_config()
            logging.info("Data ingestion has finished!!")

            logging.info("Data transformation has started.")
            data_transformation=DataTransformation()
            train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            logging.info("Data transformation has completed")

            logging.info("Model training has started")
            model_trainer=ModelTrainer()
            best_model,accuracy, prec, recall = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info("Model training has completed!! ")
            return best_model,accuracy, prec, recall


        except Exception as e:
            raise CustomException(e,sys)
    

    def create_experiment(self,experiment_name,run_name, accuracy, precision, recall, model, confusion_matrix_path = None, 
                      roc_auc_plot_path = None, run_params=None):
     
        #mlflow.set_tracking_uri("http://localhost:5000") #uncomment this line if you want to use any database like sqlite as backend storage for model
        mlflow.set_experiment(experiment_name)
        
        #dagshub.init("Hotel-Booking-Dataset-mlflow-dvc", "deepak2009thakur", mlflow=True)

       

        
        with mlflow.start_run():
            # if not run_params == None:
            #     for param in run_params:
            #         mlflow.log_param(param, run_params[param])

            
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            
            #mlflow.sklearn.log_model(model, "model")
            
            # if not confusion_matrix_path == None:
            #     mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix')
                
            # if not roc_auc_plot_path == None:
            #     mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
            
            mlflow.set_tags({"tag1":"RandomForestClassifier"})

             ## For Remote server only(DAGShub)
            remote_server_uri="https://dagshub.com/aakashsyadav1999/Healthcare.mlflow"
            mlflow.set_tracking_uri(remote_server_uri)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow

                mlflow.sklearn.log_model(
                model, "model", registered_model_name=experiment_name
                    )
            else:
                mlflow.sklearn.log_model(model, "model")


            

            # Suppress the specific warning
            #warnings.filterwarnings("ignore", message="Setuptools is replacing distutils.")

                
        print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))
        
if __name__=="__main__":

    training_pipeline = TrainingPipeline()
    best_model, accuracy, prec, recall = training_pipeline.start_model_training()

    #experiment - 1
    training_pipeline.create_experiment("Optimized_model","Random_Forest_classifier", accuracy, prec, recall, best_model, 'confusion_matrix.png', 
                     'roc_auc_plot.png')