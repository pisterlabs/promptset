
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("c://users//mccar//miniconda3//lib//site-packages")
from config import set_environment
set_environment()

# from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Classify the sentiment of this text: {text}")
model = ChatOpenAI()
# prompt = PromptTemplate(input_variables=["text"], template="Classify the sentiment of this text: {text}")
# prompt = PromptTemplate.from_template("Classify the sentiment of this text: {text}")
chain = prompt | model
print(chain.invoke({"text": "I hated that movie, it was terrible!"}))



if __name__ == "__main__":
    pass
