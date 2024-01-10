import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import openai
from dotenv import load_dotenv
import csv
from pydantic import BaseModel
from enum import Enum

load_dotenv()

# Configuration
openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0.0, model=llm_model)

# Prompt Template
prompt_template = """You are a data analyst. You have the following data columns:
{columns}
Here is the context data we have:
{content}
Here is my data in the following form:
{features}
{my_features}
Please predict the following label: {label}
Answer only the label and nothing else: """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["columns", "content", "features", "my_features", "label"]
)

class StressLevel(str, Enum):
    no_stress = "no stress"
    low_stress = "low stress"
    medium_stress = "medium stress"
    high_stress = "high stress"
    very_high_stress = "very high stress"

class NewData(BaseModel):
    title: str
    description: str
    stress: StressLevel  # ["no stress", "low stress", "medium stress", "high stress", "very high stress"]


def complexity_prediction(new_data: NewData):
    # Load the CSV data
    with open('user_history.csv', 'r') as file:
        reader = csv.reader(file)
        columns = next(reader)
        content = list(reader)

    my_features = [new_data.title, new_data.description, new_data.stress]
    predictor = PROMPT.format(columns=columns, content=content, features=columns[:-1], my_features=my_features, label=columns[-1])

    system = "You are a helpful assistant that answers human question in the most concise and to the point way as possible. Please follow the human's instructions precisely"
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=predictor),
    ]

    return chat(messages)


if __name__ == "__main__":
    mock_example = NewData(title="page is not loading", description="page is stuck on loop. After awhile I get a time out request. The server seems to be running", stress="medium stress")
    response = complexity_prediction(mock_example)
    print(response)
