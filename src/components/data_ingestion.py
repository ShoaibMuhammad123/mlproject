import os 
import sys 
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd 

## after data transformation part 
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
# using data class is only recommand if we have attributes only in class , and have no function, if have function also then use constructor technique
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")
    

class Data_ingestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df  = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Data Read as a DataFrame')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            ### Storing full data 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('Splitting data initiated')
            ## splitting data 
            train_data,test_data = train_test_split(df,test_size=.2,random_state=42)
            logging.info('Splitting data Completed')
            # storing train and test data in respective path
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Ingestion of Data is Completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj = Data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))