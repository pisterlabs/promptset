from model_pipeline import get_model_pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import create_pandas_dataframe_agent, initialize_agent
from langchain.agents import load_tools
import pandas as pd

#model_id = "tiiuae/falcon-40b"
model_name = "falcon-40b-instruct"
#model_name = "open_llama_13b"
#model_name = "falcon-40b"
#model_name = "xgen-7b-8k-base"exit()
#https://huggingface.co/TheBloke/airoboros-65B-gpt4-1.2-GGML

pipe = get_model_pipeline(model_name)
llm = HuggingFacePipeline(model_id = model_name, pipeline = pipe)

df_errors= pd.read_csv("/data/preprocessed/logs/2023-06-28/error.csv")
#df_warnings= pd.read_csv("/data/preprocessed/logs/2023-06-28/warning.csv")

agent = create_pandas_dataframe_agent(llm, df_errors, verbose=True)
#agent.prompt.template
#new_prompt = 

answer = agent.run("Which host wrote the last message according to the timestamps?")
#answer = agent.run("give me the name of one random host")
print("answer:", answer)
