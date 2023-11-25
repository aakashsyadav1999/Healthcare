import os 
import sys
import numpy as np
from dataclasses import dataclass
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse

from src.Healthcare.exception import CustomException
from src.Healthcare.logger import logging
from src.Healthcare.utils import save_object, evaluate_models


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info("Split training data and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "RandomForestClassifier": RandomForestClassifier(),
            }
            
            params={
                "RandomForestClassifier":{
                   'n_estimators' : [100],
                   'n_jobs' : [-1],
                   'max_features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                   'max_depth':[3, 4, 5, 6, 7, 9, 11]
                }
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"This is the best model{best_model}")
            

            #best_model_Score
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            predicted=best_model.predict(X_test)
            acc = accuracy_score(y_test, predicted)
            prec = precision_score(y_test, predicted,pos_label='positive',average='weighted')
            recall = recall_score(y_test, predicted,pos_label='positive',average='weighted')
            return best_model, acc, prec, recall
            print(acc)
            print(prec)
            print(recall)
            logging.log(acc,prec,recall)
                     
        except Exception as e:
            raise CustomException(e,sys)
