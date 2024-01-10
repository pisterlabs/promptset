import pickle
import yaml
import os
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory

# Load the API key from the YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
# Set the API key in the environment variables
os.environ["OPENAI_API_KEY"] = config['openai_api_key']

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question is about personal finance management and personal budgeting.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about the financial independence in the Philippines.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about personal finance management, politely inform them that you are tuned to only answer questions about the methods to achieve financial independence.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain

if __name__ == "__main__":
    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("vectorstore.index", embeddings)

    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("AI:")
        print(result["answer"])
