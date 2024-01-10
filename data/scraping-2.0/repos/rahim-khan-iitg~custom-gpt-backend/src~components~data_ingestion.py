import os
import sys
from utils.logger import logging
from utils.exceptions import CustomException
from langchain.document_loaders.csv_loader import CSVLoader
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    data_path=os.path.join("data","data.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()
    
    def initialize_data_ingestion(self):
        try:
            logging.info("data loading started")
            print(self.ingestion_config.data_path)
            loader=loader=CSVLoader(file_path=self.ingestion_config.data_path,encoding="utf-8")
            data=loader.load()
            logging.info(str(data[0:3]))
            return data
        except Exception as e:
            logging.info("error occured during data ingestion")
            raise CustomException(sys,e)