from langchain.memory import ReadOnlySharedMemory
from memory import memory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
import config
import os

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

temperature = 0

def read_first_3_rows():
    dataset_path = "dataset.csv"
    try:
        df = pd.read_csv(dataset_path)
        first_3_rows = df.head(3).to_string(index=False)
    except FileNotFoundError:
        first_3_rows = "Error: Dataset file not found."

    return first_3_rows

def get_chatresponse(string):

    dataset_first_3_rows = read_first_3_rows()

    CHAT_TEMPLATE_PREFIX = """You are an AI Assitant that can help explore datasets. offer brief and poliet smalltalk.
    First 3 rows of the dataset:"""

    DATASET = f"{dataset_first_3_rows}"

    CHAT_TEMPLATE_SUFFIX = """====    

    ====
    conversation history:
    {chat_history}
    ====
    User's New Input: {question}
    ====

    AI:"""

    CHAT_TEMPLATE = CHAT_TEMPLATE_PREFIX + DATASET + CHAT_TEMPLATE_SUFFIX
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    short_llm = ChatOpenAI(temperature=temperature)
    long_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=temperature)
    llm = short_llm.with_fallbacks([long_llm])

    template = CHAT_TEMPLATE
    chat_prompt = PromptTemplate(template=template, input_variables=["question", "chat_history"])
    chat_chain = LLMChain(prompt=chat_prompt, llm=llm, memory=readonlymemory, verbose=True)
    chat_response = chat_chain.run(string)
    return chat_response