import os 
import sys 
from dataclasses import dataclass 

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
     

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
        
        ## train_arry and test_array are output of model transformation so it will be my input to model trainer
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting training and test data') 
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],   # train features
                train_array[:,-1],    # train target
                test_array[:,:-1],    # test features 
                test_array[:,-1],     # test target 
            )
            
            # now i am creating dictionary of models 
            
            models ={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Nearest Neighbour":KNeighborsRegressor(),
                "XGBoost Regressor":XGBRegressor(),
                "Catboost Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
                
            }
            
            model_report:dict = evaluate_models(X_train=X_train,y_train = y_train,X_test=X_test,y_test=y_test,models=models)
            
            ## to get the best model score from dict
            best_model_score = max(sorted(model_report.values() ))
            
            # now to best model name 
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)
            ]
            
            # to pick up best model name
            best_model = models[best_model_name]
            
            if best_model_score <.6:
                raise CustomException('No Best Model Found')
            
            logging.info(f"Best found model on both training and testing datasets")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            
            ### to see the predicted output
            predicted = best_model.predict(X_test)
            score = r2_score(y_test,predicted)
            
            return score
        except Exception as e:
            raise CustomException(e,sys) 