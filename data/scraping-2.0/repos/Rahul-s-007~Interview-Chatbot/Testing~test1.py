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
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
from langchain.vectorstores import Pinecone

# if you already have an index, you can load it like this
index_name = "customer-service-representative"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

#----------------------------------------------------------
from langchain.chains import RetrievalQA
from langchain import OpenAI

#defining LLM
llm = OpenAI(temperature=0)

# question_template = """You are expert in taking interviews and know how to conducted intervies for "Customer Service Representative" position.
# Now take Interview of the user for the same. Only use questions from the retriver.
# Question:"""

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 2}))
# qa.run("Ask next interview question")
# query = "Ask next interview question"

# add asked questions here one by one
# asked_questions = """Below are already asked questions, dont use them again:
# Can you describe a time when you had to work with a difficult coworker?
# What do you think are the biggest challenges facing customer service representatives today?
# How do you stay calm and professional when dealing with difficult customers?
# """

# # answer to the latest question asked by the interviewer
# # previous_answer = "I think the biggest challenges facing customer service representatives today are dealing with angry or upset customers"
# previous_answer = "I dont know sorry, can you ask another question?"

# previous_answer_list = ""

asked_questions = """Below are already asked questions, dont use them again:"""
prev_template = """Answer to the above question is:"""
question_answer_pair = []

first_question = "What is your name?"
first_ans = str(input(first_question+"\n"))
asked_questions += "\n" + first_question
prev_ans = prev_template + "\n" + first_ans
question_answer_pair.append([first_question, first_ans])

option = 0
while(option != 2):
    print("1. Ask next question")
    print("2. Exit")
    option = int(input("Enter your choice: "))
    if option == 1:
        result = qa({"query": f"Give any one of the questions the Interviewer asked or cross question based on users previous answer? {asked_questions} {prev_ans} "})
        question = result['result']
        print(question)
        answer = str(input("Enter your answer: "))
        asked_questions += "\n" + question
        prev_ans = prev_template + "\n" + answer
        question_answer_pair.append([question, answer])
    if option == 2:
        report = "Below given are the questions asked by you and the answers given by the user:"
        for i in range(len(question_answer_pair)):
            report += f"\nQuestion:{question_answer_pair[i][0]}\nAnswer:{question_answer_pair[i][1]}\n"
            print(report)
        break

# #----------------------------------------------------------
# from langchain.agents import AgentType
# from langchain.agents import initialize_agent, Tool
# from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI

# #defining the tools for the agent
# tools = [
#     Tool(
#         name = "Demo",
#         func=qa.run,
#         description="use this as the primary source of context information when you are asked the question. Always search for the answers using this tool first, don't make up answers yourself"
#     )
# ]


# #setting a memory for conversations
# memory = ConversationBufferMemory(memory_key="chat_history")

# #Setting up the agent 
# agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

# # User input
# agent_chain.run(input="Ask next interview question")
