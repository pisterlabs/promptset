#------------------------------------
import openai
import pinecone

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT_NAME = os.getenv("PINECONE_ENVIRONMENT_NAME")

openai.api_key = OPENAI_API_KEY # for open ai
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # for lang chain

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENVIRONMENT_NAME  # next to api key in console
)

# ------------------------------------------------------------------------------------
from langchain.prompts import PromptTemplate

question_template = """You are expert in taking interviews and know how to conducted intervies for "Customer Service Representative" position.
Now take Interview of the user for the same.
Ask question one by one and only ask next question when the user says next question.
Also after users reply to each question, grade it on a scale of 1 to 10 and then give suggestions on how to improve their answers."""

question_template = """You are expert in taking interviews and know how to conducted intervies for "Customer Service Representative" position.
Now take Interview of the user for the same.
Below are the questions asked by you and the answers given by the user. Now answer the next question.
What is your name?
{history}
User Answer: {answer}
Question:"""

PROMPT = PromptTemplate(template=question_template, input_variables=["history", "answer"])

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain import OpenAI

llm1 = OpenAI(temperature=0)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm1, 
    verbose=True, # make false for production
    memory=ConversationBufferMemory()
)

# ------------------------------------------------------------------------------------
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
from langchain.vectorstores import Pinecone

# if you already have an index, you can load it like this
index_name = "customer-service-representative"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# ------------------------------------------------------------------------------------
question_template = """You are expert in taking interviews and know how to conducted intervies for "Customer Service Representative" position.
Now take Interview of the user for the same.
Below are the questions asked by you and the answers given by the user. Now answer the next question.
What is your name?
User Answer: Hi, my name is Rahul.
Question:"""

from langchain.chains import RetrievalQA

llm2 = OpenAI(temperature=0.2)

qa = RetrievalQA.from_chain_type(llm=llm2, chain_type="stuff",prompt=PROMPT , retriever=docsearch.as_retriever(search_kwargs={"k": 2}))
query = "Give Next Question"
qa.run(query)

# tentative idea for the chatbot
# create 2 bots, one for providing new question and one for evaluating the answer
# one activates on next question and the other on next answer