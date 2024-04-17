import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, eval_model, evaluate_models

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split Train and Test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
                )
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Support Vector Classifier":SVC(),
                "XGB Classifier": XGBClassifier(),
                "CatBoost Classifier":CatBoostClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier()
            }

            params={
                "Logistic Regression":{
                    'penalty':['l1','l2','elasticnet']
                },
                "K-Neighbors Classifier":{},
                "Decision Tree Classifier": {
                    # 'criterion':['gini', 'entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2',None],
                },
                "Random Forest Classifier":{
                    # 'criterion':['gini', 'entropy', 'log_loss'],
                    # 'max_features':['sqrt','log2',None],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Support Vector Classifier":{
                    # 'kernel':['linear','poly','rbf','sigmoid','precomputed']
                },
                "XGB Classifier":{
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Classifier":{
                    # 'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    # 'iterations': [30, 50, 100]
                },
                "Gradient Boosting Classifier":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Classifier":{
                    # 'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    # 'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # TO get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            print(best_model_name)
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            accuracy, precision, recall, f1 = eval_model(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e.sys)