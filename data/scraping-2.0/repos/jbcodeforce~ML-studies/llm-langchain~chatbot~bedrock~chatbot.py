'''
A chatbot with LLM and contextual aware embeddings.
For the chatbot we need context management, history, vector stores
This implementation uses LangChain library to chain the components.
Use a csv file with FAQ on a given subject (SageMaker).
This vector is stored in FAISS.
'''

import os
import sys


module_path = "."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper



def buildBedrockClient():
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    os.environ["BEDROCK_ENDPOINT_URL"] = "https://bedrock." + os.environ["AWS_DEFAULT_REGION"] + ".amazonaws.com/"

    return  bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )

def buildConversationWithBedrockTitanLLM(bedrock_client):
    '''
    Build a conversation chain with a Titan LLM keeping chat history.
    ConversationBufferMemory stores messages and extracts them as variable.
    use Retrieval Augmented Generation, with a vector store in memory. 
    '''
    titan_llm = Bedrock(model_id="amazon.titan-tg1-large", client=bedrock_client)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
    vectorStore = buildVectorStore(bedrock_client)
    validateAQuery(vectorStore,titan_llm)
    qa= ConversationalRetrievalChain.from_llm(
                        llm=titan_llm, 
                        retriever=vectorStore.as_retriever(search_type='similarity', search_kwargs={"k": 8}),
                        memory=memory,
                        verbose=True,
                        chain_type='stuff', # 'refine',
                        max_tokens_limit=100
                    )
    qa.combine_docs_chain.llm_chain.prompt= create_qa_with_context_template()
    return qa


def buildVectorStore(bedrock_client):
    '''
    Build in-memory vector store from external documents. This process should NOT be in this code
    but done in separate ETL process, still using LangChain, but with external persistence. The 
    Store needs to implement a retriever. 
    For embeddings, we use Titan
    '''
    print(".... build vector store from Amazon SageMaker FAQ.... takes some time")
    loader = CSVLoader("./rag_data/Amazon_SageMaker_FAQs.csv") # --- > 219 docs with 400 chars
    corpus_documents = loader.load()
    chunked_docs = CharacterTextSplitter(chunk_size=2000, 
                                 chunk_overlap=400, 
                                 separator=",").split_documents(corpus_documents)
    embeddings = BedrockEmbeddings(client=bedrock_client)
    faiss_store = FAISS.from_documents(
                        documents=chunked_docs,
                        embedding = embeddings)
    print("Done loading vector store") 
    return faiss_store


def validateAQuery(vectorStore,llm):
    wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorStore)
    print_ww(wrapper_store_faiss.query("R in SageMaker", llm=llm))

'''
Build a custom prompt with chat history to rephrase the question.
'''
def create_qa_with_history_template():
    _template = """{chat_history}

Answer only with the new question.
How would you ask the question considering the previous conversation: {question}
Question:"""
    return PromptTemplate.from_template(_template)

'''
Build a custom prompt with context and question.
'''
def create_qa_with_context_template():
    _template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}

    Question: {question}
    ANSWER:"""
    return PromptTemplate.from_template(_template)

class ChatUX:
    """ A chat UX using IPWidgets
    """
    def __init__(self, qa, retrievalChain = False):
        self.qa = qa
        self.name = "A_Basic_ChatBot"
        self.retrievalChain = retrievalChain


    def start_chat(self):
        print("Starting chat bot")
        prompt="Hello, I am a chatbot. What's your name?"
        finish=False
        chat_history = []
        while not finish:
            prompt=input("You (q to quit): ")
            if prompt=='q' or prompt=='quit' or prompt=='Q':
                finish=True
            else:
                try:
                    if self.retrievalChain:
                        result = self.qa.run({'question': prompt })
                    else:
                        result = self.qa.run({'input': prompt , 'history':chat_history})
                except Exception as e:
                    print_ww(e)
                    result = "No answer"
                print_ww(f"AI: {result}")
        print("Thank you , that was a nice chat !!")


if __name__ == "__main__":
    print("initialize the bot")
    bedrock_client= buildBedrockClient()
    
    conversation = buildConversationWithBedrockTitanLLM(bedrock_client)
    
    # Start the chatbot
    chat = ChatUX(conversation,retrievalChain=True)
    chat.start_chat()   