from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI

class GPTModel:
   def __init__(self, directory_path):

       # set maximum input size
      self.max_input_size = 4096

      # set number of output tokens
      self.num_outputs = 2000

      # set maximum chunk overlap
      self.max_chunk_overlap = 20

      # set chunk size limit
      self.chunk_size_limit = 600 

      self.directory_path = directory_path


   def construct_index(self): 

      llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=self.num_outputs))

      prompt_helper = PromptHelper(self.max_input_size, self.num_outputs, self.max_chunk_overlap, chunk_size_limit=self.chunk_size_limit)

      documents = SimpleDirectoryReader(self.directory_path).load_data()

      index = GPTSimpleVectorIndex(
         documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
      )

      index.save_to_disk('gptModel.json')