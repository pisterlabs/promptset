# Import necessary modules
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.llms import OpenAI


load_dotenv('.env')

### Example with a chat model

# Initialize language model
# llm = OpenAI(model_name="text-davinci-003", temperature=0)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

template = "You are an assistant that helps users find information about movies."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = "Find information about the movie {movie_title}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

with get_openai_callback() as cb:
    response = llm(chat_prompt.format_prompt(movie_title="Interstellar").to_messages())
    print(response.content)
    print(cb)


### Example with a non chat model
llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

with get_openai_callback() as cb:
    print(chain.run("what is the meaning of life?"))
    print(cb)