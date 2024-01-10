import os
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex,\
                      LLMPredictor, PromptHelper
from langchain import OpenAI
from dotenv import load_dotenv


load_dotenv()

def create_index(path):
  max_input = 4096
  tokens = 200
  chunk_size = 600 # for LLM, we need to define chunk size
  max_chunk_overlap = 20
  
  #define prompt
  prompt_helper = PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size)
  
  #define LLM — there could be many models we can use, but in this example, let’s go with OpenAI models
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-curie-001", max_tokens=tokens))
  
  #load data — it will take all the .txtx files, if there are more than 1
  docs = SimpleDirectoryReader(path).load_data()

  #create vector index
  vector_index = GPTSimpleVectorIndex(documents=docs,llm_predictor=llm_predictor,prompt_helper=prompt_helper)
  vector_index.save_to_disk('chat/data/vector_index.json')
  return vector_index

if __name__ == "__main__":
    vector_index = create_index(os.path.dirname("chat/data/"))
