import os
from getpass import getpass

import streamlit as st 
from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DeepLake
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

if os.environ.get('OPENAI_API_KEY') is None:
    os.environ['OPENAI_API_KEY'] = getpass('OpenAI API key')

if os.environ.get('ACTIVELOOP_TOKEN') is None:
    os.environ['ACTIVELOOP_TOKEN'] = getpass('Activeloop Token:')

deeplake_account_name = "nithyash"
dataset_name = "llama-code"
root_dir = '/Users/nithyashreemanohar/opt/anaconda3/envs/nlp/lib/python3.7/site-packages/llama'

def upload_dataset(deeplake_account_name, dataset_name, root_dir):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.py') and '/.venv/' not in dirpath:
                try: 
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    docs.extend(loader.load_and_split())
                except Exception as e: 
                    pass

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = DeepLake.from_documents(texts, embeddings, dataset_path=f"hub://{deeplake_account_name}/{dataset_name}")

def load_dataset_retriever(deeplake_account_name, dataset_name):
    embeddings = OpenAIEmbeddings()
    db = DeepLake(dataset_path=f"hub://{deeplake_account_name}/{dataset_name}", read_only=True, embedding_function=embeddings)

    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 20
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 20

    return db, retriever

def load_qa_model(retriever):
    model = ChatOpenAI(model='gpt-3.5-turbo') # 'ada' 'gpt-3.5-turbo' 'gpt-4',
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

    return qa

def get_function_docs(function_name, qa):
  parameters_text = "What are the parameters for the function {function}? Answer like API reference documentation. Say 'Parameters:' and then list the parameters and their description."
  outputs_text = "What is the output of the function {function}? Answer like API reference documentation. Say 'Outputs:' and then list the outputs. Say 'Outputs: None' if there are no outputs."
  description_text = "What does the function {function} do? Answer like API reference documentation. Say 'Description:' and then answer in 2 to 5 lines."

  parameters_template = PromptTemplate(
      input_variables=["function"],
      template=parameters_text
  )
  outputs_template = PromptTemplate(
      input_variables=["function"],
      template=outputs_text
  )
  description_template = PromptTemplate(
      input_variables=["function"],
      template=description_text
  )

  templates = [parameters_template, outputs_template, description_template]

  docstring = function_name + "  \n  \n"

  chat_history =  []
  for template in templates:
    result = qa({"question": template.format(function=function_name), "chat_history": chat_history})
    docstring += result["answer"]
    docstring += "  \n  \n"

  return docstring

# import deeplake
# print(deeplake.exists("hub://{deeplake_account_name}/{dataset_name}", creds="ENV", token=os.environ['ACTIVELOOP_TOKEN']))

# uncomment this line to create the dataset, comment if dataset already exists
# upload_dataset(deeplake_account_name, dataset_name, root_dir)

db, db_retriever = load_dataset_retriever(deeplake_account_name, dataset_name)

qa = load_qa_model(db_retriever)

st.title("Question Answers for Code Understanding")

user_question = st.text_input(
    "Enter function name to explain : ",
    placeholder = "Function name here ",
)

if st.button("Tell me about it", type="primary"):
    function_docstring = get_function_docs(user_question, qa)
    print(function_docstring)
    st.success(function_docstring)