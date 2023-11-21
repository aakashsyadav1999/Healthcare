import os
import sys
import pandas as pd

from src.Healthcare.exception import CustomException
from src.Healthcare.logger import logging
from src.Healthcare.utils import save_object,load_obj



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        no_of_adults: int,
        no_of_children: int,
        no_of_weekend_nights: int,
        no_of_week_nights: int,
        type_of_meal_plan: str,
        required_car_parking_space: int,
        room_type_reserved: str,
        arrival_year: int,
        arrival_month: int,
        lead_time:int,
        arrival_date: int,
        market_segment_type:str,
        repeated_guest:int,
        no_of_previous_cancellations:int,
        no_of_previous_bookings_not_canceled:int,
        avg_price_per_room:float,
        no_of_special_requests:int):




        self.no_of_adults = no_of_adults

        self.no_of_children = no_of_children

        self.no_of_weekend_nights = no_of_weekend_nights

        self.no_of_week_nights = no_of_week_nights

        self.type_of_meal_plan = type_of_meal_plan

        self.required_car_parking_space = required_car_parking_space

        self.room_type_reserved = room_type_reserved

        self.arrival_year = arrival_year

        self.arrival_month = arrival_month

        self.lead_time = lead_time

        self.arrival_date = arrival_date

        self.market_segment_type = market_segment_type

        self.repeated_guest = repeated_guest

        self.no_of_previous_cancellations = no_of_previous_cancellations

        self.no_of_previous_bookings_not_canceled = no_of_previous_bookings_not_canceled

        self.avg_price_per_room = avg_price_per_room

        self.no_of_special_requests = no_of_special_requests



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "no_of_adults": [self.no_of_adults],
                "no_of_children": [self.no_of_children],
                "no_of_weekend_nights": [self.no_of_weekend_nights],
                "no_of_week_nights": [self.no_of_week_nights],
                "type_of_meal_plan": [self.type_of_meal_plan],
                "required_car_parking_space": [self.required_car_parking_space],
                "room_type_reserved": [self.room_type_reserved],
                "arrival_year": [self.arrival_year],
                "arrival_month": [self.arrival_month],
                "lead_time": [self.lead_time],
                "arrival_date": [self.arrival_date],
                "market_segment_type": [self.market_segment_type],
                "repeated_guest": [self.repeated_guest],
                "no_of_previous_cancellations": [self.no_of_previous_cancellations],
                "no_of_previous_bookings_not_canceled": [self.no_of_previous_bookings_not_canceled],
                "avg_price_per_room": [self.avg_price_per_room],
                "no_of_special_requests": [self.no_of_special_requests],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)