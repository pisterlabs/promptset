import pandas as pd
from pandasai import PandasAI
from langchain import HuggingFacePipeline
import torch
from transformers import AutoTokenizer
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain import PromptTemplate, LLMChain

# Huggingface login
from huggingface_hub.hf_api import HfFolder
import os

token = os.environ.get("HUGGINGFACE_TOKEN", None)
HfFolder.save_token(token)


# Sample DataFrame
df = pd.DataFrame(
    {
        "country": [
            "United States",
            "United Kingdom",
            "France",
            "Germany",
            "Italy",
            "Spain",
            "Canada",
            "Australia",
            "Japan",
            "China",
        ],
        "gdp": [
            19294482071552,
            2891615567872,
            2411255037952,
            3435817336832,
            1745433788416,
            1181205135360,
            1607402389504,
            1490967855104,
            4380756541440,
            14631844184064,
        ],
        "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12],
    }
)

model_path = "tiiuae/falcon-7b"
# model_path = "bigcode/starcoder"
#  GPT neo
tokenizer = AutoTokenizer.from_pretrained(model_path)
# model_path :  "EleutherAI/gpt-neo-2.7B"

llm = HuggingFacePipeline.from_model_id(
    # model_id="bigscience/bloom-1b7",
    model_id=model_path,
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 3000,
        # "trust_remote_model": True,
        # "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "top_k": 10,
        "num_return_sequences": 1,
    },
    model_kwargs={
        "use_auth_token": token,
        # "trust_remote_model": True,
        "temperature": 0.3,
        # "max_length": 30000,
        # "max_new_tokens": 300,
        "early_stopping": True,
        # "max_time": 30,
        "max_length": 200,
        "do_sample": True,
        "top_k": 10,
        "num_return_sequences": 1,
        # "eos_token_id": tokenizer.eos_token_id,
    },
    device=0,
)

template = """This is my complete, step-by-step recipe for a delicious {dish}: 
First """
prompt = PromptTemplate(template=template, input_variables=["dish"])

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# question = "What is electroencephalography?"
dish = "death-by-chocolate cake"

print(llm_chain.run(dish))


agent = create_pandas_dataframe_agent(
    llm, df, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

agent.run("how many rows are there?")
