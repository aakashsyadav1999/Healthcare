import os
import sys
import pandas as pd

from src.Healthcare.exception import CustomException
from src.Healthcare.logger import logging
from src.Healthcare.utils import load_object,save_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Age: int,
        Gender: str,
        Blood_Type: str,
        Medical_Condition: str,
        Doctor: str,
        Hospital: str,
        Insurance_Provider: str,
        Room_Number: int,
        Admission_Type: str,
        Medication:str
    ):




        self.Age = Age

        self.Gender = Gender

        self.Blood_Type = Blood_Type

        self.Medical_Condition = Medical_Condition

        self.Doctor = Doctor

        self.Hospital = Hospital

        self.Insurance_Provider = Insurance_Provider

        self.Room_Number = Room_Number

        self.Admission_Type = Admission_Type

        self.Medication = Medication



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Blood_Type": [self.Blood_Type],
                "Medical_Condition": [self.Medical_Condition],
                "Doctor": [self.Doctor],
                "Hospital": [self.Hospital],
                "Insurance_Provider": [self.Insurance_Provider],
                "Room_Number": [self.Room_Number],
                "Admission_Type": [self.Admission_Type],
                "Medication": [self.Medication]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)