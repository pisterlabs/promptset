import json
import pandas as pd
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import openai
from langchain.output_parsers import CommaSeparatedListOutputParser

# Open the file for reading
with open('topic_modeling/topic_info.json', 'r') as file:
    data = json.load(file)
# 使用os.getenv访问你的环境变量
openai.api_key = os.getenv("openai_api_key")
openai.api_base = os.getenv("openai_api_base")

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
# One shot learning, prompt
base_prompt = """
根據前面bertopic輸出的topic，推斷出機率最高（從高到低排）的五種job title
"""

# full prompt
prompt = PromptTemplate(
    template="{question} \n {base_prompt} \n{format_instructions}",
    input_variables=["question"],
    partial_variables={
        "base_prompt": base_prompt,
        "format_instructions": format_instructions,
    },
)

# Create chat model
chat = ChatOpenAI(
    temperature=0, openai_api_key=openai.api_key, openai_api_base=openai.api_base
)

topic_data = pd.DataFrame.from_dict(data, orient='index')
topic_data = topic_data.reset_index().rename(columns={'index': 'topic'})

for key, value in data.items():
    _input = prompt.format(question=str(value))
    output = chat.predict(_input)
    print(output)
    topic_data.loc[int(key) + 1,"job_title"] = str(output)

topic_data.to_csv("topic_modeling/topic_seed.csv")
