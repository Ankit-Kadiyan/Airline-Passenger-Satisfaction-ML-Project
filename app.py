import sys
from src.logger import logging
from src.exception import CustomException
# from src.componenets.data_ingestion import DataIngestionConfig
from src.componenets.data_ingestion import DataIngestion
# from src.componenets.data_transformation import DataTransformationConfig
from src.componenets.data_transformation import DataTransformation
# from src.componenets.model_trainer import ModelTrainerConfig
from src.componenets.model_trainer import ModelTrainer

if __name__=="__main__":
    logging.info("The execution has started.")

    try:
        # data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        # data_transform_config=DataTransformationConfig()
        data_transform=DataTransformation()
        train_arr, test_arr, _=data_transform.initiate_data_transformation(train_data_path,test_data_path)

        # Model Training
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
