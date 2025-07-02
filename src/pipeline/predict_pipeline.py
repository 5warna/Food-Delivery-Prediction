import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
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
    def __init__(self,
                 Distance_km: float,
                 Weather: str,
                 Traffic_Level: str,
                 Time_of_Day: str,
                 Vehicle_Type: str,
                 Preparation_Time_min: int,
                 Courier_Experience_yrs: float):

        self.Distance_km = Distance_km
        self.Weather = Weather
        self.Traffic_Level = Traffic_Level
        self.Time_of_Day = Time_of_Day
        self.Vehicle_Type = Vehicle_Type
        self.Preparation_Time_min = Preparation_Time_min
        self.Courier_Experience_yrs = Courier_Experience_yrs

    def get_data_as_dataframe(self):
        """
        Returns user input as a pandas DataFrame (1-row) for model prediction.
        """
        try:
            data_dict = {
                "Distance_km": [self.Distance_km],
                "Weather": [self.Weather],
                "Traffic_Level": [self.Traffic_Level],
                "Time_of_Day": [self.Time_of_Day],
                "Vehicle_Type": [self.Vehicle_Type],
                "Preparation_Time_min": [self.Preparation_Time_min],
                "Courier_Experience_yrs": [self.Courier_Experience_yrs]
            }

            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
