
import numpy as np # Linear Algebra
import pandas as pd # Data Processing 
# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from getpass import getpass
import openai,os

import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFramePreprocessor:
    def __init__(self, file_path):
  
        self.file_path = file_path
        self.df = None

    def load_data(self):

        try:
            self.df = pd.read_csv(self.file_path)
            shape  = self.df.shape
            logger.info("shape of Data %s", shape)
            logger.info("Data loaded successfully from %s", self.file_path)
        except FileNotFoundError as e:
            logger.error("File not found: %s", self.file_path)
            raise e

    def remove_duplicates(self):
   
        self.df = self.df.drop_duplicates()
        logger.info("Duplicates removed")
        
    
    def prepare_data_for_analysis(self):
        self.df.drop(columns=['retweetedTweet' , 'quotedTweet','media','outlinks'], inplace=True)
        logger.info("remove unnecessary columns ")
        logger.info("set index to timestep ")
        self.df.set_index('timestamp',inplace=True)


    def fill_missing_values(self, column, strategy='mean'):

        if strategy == 'mean':
            self.df[column].fillna(self.df[column].mean(), inplace=True)
        elif strategy == 'median':
            self.df[column].fillna(self.df[column].median(), inplace=True)
        elif strategy == 'mode':
            self.df[column].fillna(self.df[column].mode().iloc[0], inplace=True)
        else:
            self.df[column].fillna(strategy, inplace=True)
        logger.info("Missing values in column %s filled using strategy %s", column, strategy)

    def feature_engineering(self):
        self.df = self.df.drop(['created_at'] , axis=1)
        self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d %H:%M:%S%z')
        self.df['timestamp'] = self.df['date'].dt.strftime("%Y-%m-%d %H:%M:%S")
        self.df = self.df.drop(columns=['date'])

    def preprocess_data(self):
   
        self.remove_duplicates()
        self.feature_engineering()
        self.prepare_data_for_analysis()
    

    def get_preprocessed_data(self):
  
        return self.df
    
    def save_cleaned_data(self):
        
        self.df.to_csv('../data/cleaned_data.csv' , index=True)

    def display_preprocessed_data(self):
        """
        Display the preprocessed DataFrame.
        """
        print(self.df.head())

def main(file_path=None):

    parser = argparse.ArgumentParser(description='Preprocess CSV data')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing the data')
    args = parser.parse_args()
    
    if args.file_path:
        
        preprocessor = DataFramePreprocessor(args.file_path)
    else:
        preprocessor = DataFramePreprocessor(file_path)
    preprocessor.load_data()
    preprocessor.preprocess_data()
    preprocessor.save_cleaned_data()
    preprocessor.display_preprocessed_data()

    data = preprocessor.get_preprocessed_data()
    print(data.index)

    return data

if __name__ == "__main__":
    main()
