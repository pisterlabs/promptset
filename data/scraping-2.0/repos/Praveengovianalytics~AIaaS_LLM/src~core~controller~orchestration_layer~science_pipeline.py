import os
import pandas as pd
from langchain.schema import OutputParserException


class DataPipeline:
    def __init__(self, directory_path):
        self.allowed_extensions = ['.csv', '.json', '.xlsx', '.xml', '.xls']
        self.files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                      not f.startswith('.') and any(f.endswith(ext) for ext in self.allowed_extensions)]

    def load_pd(self, file_path):

        if (file_path.endswith('.csv')):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif (file_path.endswith('.json')):
            df = pd.read_json(file_path)

        elif (file_path.endswith('.xml')):
            df = pd.read_xml(file_path)
        else:
            df = pd.DataFrame()
        return df

    def process(self):
        if len(self.files) == 1:
            return self.load_pd(self.files[0])
        else:
            df = []
            for i in self.files:
                df.append(self.load_pd(i))
            return df
