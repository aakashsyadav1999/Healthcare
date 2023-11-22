import sys
import os
import pandas as pd
import numpy as np


from dataclasses import dataclass


from src.Healthcare.exception import CustomException
from src.Healthcare.logger import logging
from src.Healthcare.utils import save_object



from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
        
    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()

    def feature_selection(self,data):
        """
        This function is responsible to get those feature which is important for the model
        """
        try:
            logging.info("Entered in feature selection method")
            data = data.drop(['Date_of_Admission','Billing_Amount','Discharge_Date','Name'], axis =1)
            data = data.reset_index(drop=True)
            return data
        except Exception as e:
            raise CustomException(e,sys)    
    
    
    
    
    #Features Encoding
    def features_encoding(self, x, y):
        """
        This function will return encoded data of categorical column based on Target Encoding
        """

        try:
            logging.info("Entered in data encoding method with Target Encoding")
            encoder = TargetEncoder()
            encoder = encoder.fit(x, y)
            #encoded_data = encoder.transform(x)

            logging.info("Encoding completed!!")
            return encoder
        
        except Exception as e:
            raise CustomException(e,sys)
        
    # Standardization
    def standardization(self):
        """
        
        This function is responsible for scaling the data
        
        """

        try:
            logging.info("Scale method has been started")
            scaler = StandardScaler()
            
            return scaler
        
        except Exception as e:
            raise CustomException(e,sys)
        
     # Start Transformation    
    def get_data_transformation_object(self,x,y):

        try:
            cat_cols=[
                'Gender',
                'Blood_Type',
                'Medical_Condition',
                'Doctor',
                'Hospital',
                'Insurance_Provider',
                'Admission_Type',
                'Medication',
            ]

            num_cols=[
                'Age',
                'Room_Number'
            ]

            #steps for target encoding
            encoder = TargetEncoder()
            #encode = encoder.fit(x,y)

            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())

            ])
            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("target_encoding",encoder),
            ("StandardScalar", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{cat_cols}")
            logging.info(f"Numerical Columns:{num_cols}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_cols),
                    ("cat_pipeline",cat_pipeline,cat_cols)
                ]

            )
            
            preprocessor.fit(x,y=y)
            
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responisble for data transformation
        """

        try:
            logging.info("Entered into data transformation method")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            logging.info("Read test and train data completed")
            
            target_column_name='Test_Results'

            # outliers removals in train and test data
            # logging.info("Removal of outliers in train and test dataset")
            # train_df = self.outlier_removal(train_df)
            # test_df =  self.outlier_removal(test_df)
            

            # feature selection in train and test data
            logging.info("Feature selection in train and test dataset")
            train_df = self.feature_selection(train_df)
            test_df = self.feature_selection(test_df)

            # divide the train dataset to independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
                   
            print(input_features_train_df)
            print(target_feature_train_df)
            # divide the test dataset to independent and dependent feature

            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying Target encoding on train and test dataset")
            encoder_train = self.features_encoding(input_features_train_df,target_feature_train_df)
            encoder_test= self.features_encoding(input_features_test_df,target_feature_test_df)
            input_features_encoded_train_df = encoder_train.transform(input_features_train_df)
            input_feature_encoded_test_df = encoder_test.transform(input_features_test_df)
            
            #print(input_features_encoded_train_df)
            #print(input_feature_encoded_test_df)

            # logging.info("Applying oversampling technique-SMOTE only in training dataset")
            # input_features_sampled_train_df, target_feature_sampled_train_df = self.oversmapling_smote(input_features_encoded_train_df,target_feature_train_df)

            logging.info("Scaling of the training and test dataset")
            scaler = self.standardization()
            input_features_scaled_train_arr = scaler.fit_transform(input_features_encoded_train_df)
            input_feature_scaled_test_arr = scaler.transform(input_feature_encoded_test_df)

            print(input_features_scaled_train_arr)
            print(input_feature_scaled_test_arr)


            #creating preprocessing obj so while prediction we can preprocess
            preprocessing_obj = self.get_data_transformation_object(input_features_train_df,target_feature_train_df)


            train_arr = np.c_[
                input_features_scaled_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_scaled_test_arr, np.array(target_feature_test_df)]


            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )
            

            logging.info(f"data transformation completed!!")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
        
        except Exception as e:
            raise CustomException(e,sys)