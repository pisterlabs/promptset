import os
import pathlib

from langchain import OpenAI
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper


class ExtDataIndex:
    def __init__(self):
        self.data_loc = 'data'
        self.model_name = "gpt-3.5-turbo"
        self.index_name = f"index_{self.model_name}.json"
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.override_latest_index_check = os.environ.get('OVERRIDE_INDEX_CHECK', None)
        if self.override_latest_index_check is not None:
            self.override_latest_index_check = eval(self.override_latest_index_check)
        self.index = self.load_index()

    def check_is_data_source_updated(self):
        if self.override_latest_index_check:
            print("Skipping latest index check")
            return False

        idx_f = pathlib.Path(f"{self.root}/{self.index_name}")
        data_f = pathlib.Path(f"{self.root}/{self.data_loc}")

        data_modified_time = data_f.stat().st_mtime
        idx_modified_time = idx_f.stat().st_mtime

        if idx_modified_time < data_modified_time:
            print("Data source has been updated. Creating new index")
            return True
        else:
            print("Data source has not been updated")
            return False

    def query(self, query_str):
        resp = self.index.query(query_str, mode='default')
        print(f'Question: {query_str}')
        print(resp)
        return resp

    def load_index(self):
        idx_loaded = False
        idx_exists = os.path.exists(f"{self.root}/{self.index_name}")
        data_path = f"{self.root}/{self.data_loc}"

        data_dir_contents = os.listdir(data_path)

        if not data_dir_contents:
            print("Please add a data source")
            raise Exception("No data source present")

        if idx_exists:
            idx_updated = self.check_is_data_source_updated()
            if not idx_updated:
                print("Index loaded from disk")
                index = GPTSimpleVectorIndex.load_from_disk(f"{self.index_name}")
                idx_loaded = True

        if not idx_loaded:
            index = self.build_index()

        return index

    def build_index(self):
        # define prompt helper
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_output = 256
        # set maximum chunk overlap
        max_chunk_overlap = 20
        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=self.model_name))

        documents = SimpleDirectoryReader(f'{self.data_loc}').load_data()

        index = GPTSimpleVectorIndex(
            documents,
            llm_predictor=llm_predictor,
            prompt_helper=prompt_helper,
        )

        print("New index created and saved to disk")
        index.save_to_disk(f"{self.index_name}")
        return index


