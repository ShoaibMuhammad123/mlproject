import sys
import os 
from dataclasses import dataclass 
from src.logger import logging
from src.exception import CustomException

import pandas as pd 
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl') 
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    # this is for perrforming all the transformation like scaling , converting categorical into numericcal
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation based on different type of data
        
        """
        try:
            
            ## separating numerical and categorical features 
            logging.info('Data Transformation Initiated')
            numerical_columns = ['writing score','reading score']
            
            categorical_columns = [
                        'gender', 
                        'race/ethnicity',
                        'parental level of education',
                        'lunch', 
                        'test preparation course'
                    ]
            
            ## now we create two pipelines one is for numerical features and the other is on categorical features
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            cat_pipeline =Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('standardscaling',StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Numerical Columns: {numerical_columns}')  
            logging.info(f'Categorical Columns: {categorical_columns}')  
            
            
            ## now we combine both pipeline with column_transformer
            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline',numerical_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    
    ## Now i am starting data transformation inside this function 
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path) 
            test_df = pd.read_csv(test_path) 
            
            logging.info('Read train and test data completed.')
            
            
            logging.info('Obtaining preprocessing Object')
            
            preprocessing_obj = self.get_data_transformer_object()
            
            ## target column name 
            target_column_name = "math score"
            numerical_columns = ['writing score','reading score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )
            
            input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr =preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            
            logging.info(f'Saved Preprocessing Object.')
            
            ## For Saving the pickle file only
            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys) 
        
        