import os
import sys

from src.Healthcare.exception import CustomException
from src.Healthcare.logger import logging

import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object (file_path,obj):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
    

def load_obj (file_path):

    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    

    