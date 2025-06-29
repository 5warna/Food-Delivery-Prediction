import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.target_column = "Delivery_Time_min"

        self.numerical_columns = [
            "Distance_km", 
            "Preparation_Time_min", 
            "Courier_Experience_yrs", 
            "Delivery_Time_min"  
        ]
        self.categorical_columns = [
            "Weather",
            "Traffic_Level",
            "Time_of_Day",
            "Vehicle_Type",
        ]

    def remove_outliers_iqr(self, df, columns, factor=1.5):
        try:
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - factor * IQR
                upper = Q3 + factor * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        try:
            num_features = [col for col in self.numerical_columns if col != self.target_column]
            cat_features = self.categorical_columns

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns used: {num_features}")
            logging.info(f"Categorical columns used: {cat_features}")

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, num_features),
                ("cat_pipeline", cat_pipeline, cat_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(" Train and test data loaded successfully.")

            # Apply outlier removal
            outlier_cols = [col for col in self.numerical_columns if col != self.target_column]
            train_df = self.remove_outliers_iqr(train_df, outlier_cols)
            test_df = self.remove_outliers_iqr(test_df, outlier_cols)

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[self.target_column])
            target_feature_train_df = train_df[self.target_column]

            input_feature_test_df = test_df.drop(columns=[self.target_column])
            target_feature_test_df = test_df[self.target_column]

            # Create transformer
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Merge with target
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Save preprocessor
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(" Preprocessing object saved.")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)