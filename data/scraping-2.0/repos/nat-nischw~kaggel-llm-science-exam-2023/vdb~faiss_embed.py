"""
modify code from notebook @TeetouchQQ
"""
from pathlib import Path
from datasets import load_dataset
import gc
import ctypes
import pandas as pd
from FlagEmbedding import FlagModel
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import DataFrameLoader
import numpy as np

class FaissEmbed:
    def __init__(self, data_dir, model_name, max_data=9999999, batch_size=620000):
        self.data_dir = data_dir
        self.data = None
        self.subset_data = None
        self.model_name = model_name
        self.model = None
        self.max_data = max_data
        self.batch_size = batch_size
    
    def load_data_from_parquet(self):
        files = list(map(str, Path(self.data_dir).glob("*.parquet")))
        files.remove(f'{self.data_dir}/wiki_2023_index.parquet')
        ds = load_dataset("parquet", data_files=files)
        self.data = ds['train'].to_pandas()
        del ds
        self._cleanup_memory()
    
    def _cleanup_memory(self):
        libc = ctypes.CDLL("libc.so.6")
        _ = gc.collect()
        libc.malloc_trim(0)
    
    def subset_data(self):
        self.subset_data = self.data.head(self.max_data)
    
    def initialize_model(self):
        self.model = FlagModel(self.model_name, 
                               query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                               use_fp16=True, normalize_embeddings=True)
    
    def process_batches(self):
        for i in range(0, len(self.subset_data), self.batch_size):
            batch = self.subset_data.iloc[i:i+self.batch_size]
            # TODO: Add code to process each batch with the model
    
    def run_pipeline(self):
        self.load_data_from_parquet()
        self.subset_data()
        self.initialize_model()
        self.process_batches()

if __name__ == "__main__":
    pipeline = FaissEmbed(data_dir="wikipedia-20230701", model_name='BAAI/bge-small-en')
    pipeline.run_pipeline()
