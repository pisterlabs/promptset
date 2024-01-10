import textwrap
import pickle
import langchain
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, create_csv_agent, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, LLMChain, PromptTemplate, LLMMathChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OpenAIEmbeddings
import regex as re

import os
os.environ["OPENAI_API_KEY"] = ""


########################################################################################################

llm = OpenAI(temperature=0)

### Creating memory for the LLM

memory = ConversationSummaryBufferMemory(
        memory_key="chat_history", llm=llm, max_token_limit=100, return_messages=True
)


new_template = '''AI is a doctor that answers correctly and tells when it does not know the answer and does not hallucinate. If it does not know the answer, it uses tools mentioned below. Assistant will never tell the user to consult other doctors. 
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions.
Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

'''
#########################################################################################################


### Code to extract text from files to form a Vector DataBase

# loader = DirectoryLoader('resources', glob="./*.pdf", loader_cls=PyPDFLoader)
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)



# with open('doc_embedding.pickle', 'wb') as pkl:
#     pickle.dump(embedding, pkl)

### Loading the Embeddings for database search 

# with open('gpt/doc_embedding.pickle', 'rb') as pkl:
#     embedding = pickle.load(pkl)
    
embedding = OpenAIEmbeddings()

    
persist_directory = 'gpt/db'

# Creating DataBase/ Retriever 

# vectordb = Chroma.from_documents(documents=texts,
#                                  embedding=embedding,
#                                  persist_directory=persist_directory)

# vectordb.persist()

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#####################################################################################################

### Not rquired in the project, only for debugging 

def wrap_text_preserve_newlines(text, width=120):
  lines = text.split('\n')
  wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
  wrapped_text = '\n'.join(wrapped_lines)
  return wrapped_text


########################################################################################################

### Creating QA chain for Database search

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever)
                                  # return_source_documents=True)

### Creating Summarization chain 

summarize_template = """Being a medical professional, write a concise bullet point summary do not leave out any facts or opinions:
{text}
Summary: """

summarize_prompt = PromptTemplate(template=summarize_template, 
                        input_variables=["text"])

summarize_chain = load_summarize_chain(llm, 
                             chain_type="stuff", # refine, map-reduce, stuff, use refine for documents
                             prompt=summarize_prompt)


### Used to search within a hospitals internal database, for demonstration purposes I've just used a simple csv file

csv_agent = create_csv_agent(OpenAI(temperature=0), 
                         'gpt/medical_records.csv',
                         verbose=True,
                             max_iterations=5,memory=memory)
######################################################################################################

### Extra processing for the tools defined above

def duck_wrapper(input_text):
  search_results = DuckDuckGoSearchRun()(f"{input_text}")
  return search_results

def search(input_text):
  input_text2 = qa_chain.run(input_text)
  input_text3 = duck_wrapper(input_text+" "+input_text2)
  print("INPUT TEXT: ",input_text)
  return "Answer 1: \n"+input_text2+" \nAnswer 2: \n"+input_text3+"\nSummarize the above 2 answers"


llm_math = LLMMathChain.from_llm(llm, verbose=True)

def csv_agent_function(input_text):
    answer = csv_agent.run(input_text)
    answer+=" . Repeat everything before this sentence. You don't need to use any tools now."
    return answer


tools = [
    Tool(
        name = "Math Tool",
        func=llm_math.run,
        description="Useful when you need to do math"
    ),
    # Tool(
    #     name = "Search WebMD",
    #     func=duck_wrapper,
    #     description="useful for when you need to find answers about other non medical questions"
    # ),
    Tool(
        name = "Search",
        func=search,
        description="Best option to find out answers regarding medical, clinical or medicine questions"
    ),   
    Tool(
        name = "Document Summarizer",
        func=summarize_chain.run,
        description="Useful when you need to summarize reports, documents, paragraphs"
    ),
    Tool(
        name = "Database Reader",
        func=csv_agent_function,
        description="Write python code to retrieve data from the hospital or its patients. Data contains patient names, date of admission, medicines, gender, medical conditions, etc"
    ),
]