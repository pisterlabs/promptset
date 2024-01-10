import os

from langchain import HuggingFaceTextGenInference
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings

# import your OpenAI key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

user_info = '''

total savings are 316896 rupees. 
initial balance was 30000 rupees. 

Savings delta (calculated by taking difference of balance at start of the month and end of the month):
January: 41910 rupees
February: 38391 rupees
March: 31754 rupees
April: 38526 rupees
May: 36632 rupees
June: 26646 rupees
July: 19951 rupees
August: 25208 rupees
September: 27874 rupees

Savings to Income ratio:
January: 0.49
February: 0.38
March: 0.45
April: 0.5
May: 0.36
June: 0.50
July: 0.27
August: 0.33
September: 0.43

Savings rate:
January: 0.15
February: 0.16
March: 0.15
April: 0.23
May: 0.28
June: 0.25
July: 0.24
August: 0.43
September: 0.92
'''


# Set up knowledge base
def setup_knowledge_base():
    """
    We assume that the product knowledge base is simply a text file.
    """
    # load product catalog
    product_catalog = user_info

    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_text(product_catalog)

    llm = ChatOpenAI(openai_api_key="EMPTY",
                     openai_api_base="http://localhost:8000/v1",
                     model='meta-llama/Llama-2-7b-chat-hf',
                     temperature=0.6)
    # llm = VLLMOpenAI(
    #     openai_api_key="EMPTY",
    #     openai_api_base="http://localhost:8000/v1/chat",
    #     model_name='meta-llama/Llama-2-7b-chat-hf',
    #     temperature=0.7
    # )
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name="user-account-savings-knowledge-base"
    )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


if __name__ == '__main__':
    print(len(user_info))
    # llm = ChatOpenAI(openai_api_key="EMPTY",
    #                  openai_api_base="http://localhost:8000/v1",
    #                  model='meta-llama/Llama-2-7b-chat-hf',
    #                  temperature=0)
    inference_server_url_local = "http://127.0.0.1:8080"

    llm = HuggingFaceTextGenInference(
        inference_server_url=inference_server_url_local,
        temperature=0.1,
    )
    SYS_PROMPT = f'''
    <s>[INST] <<SYS>>
    You have to act as a personal finance assistant. With the given information, try to answer the user query.
    Today's date is 10th Oct 2023. 
    <</SYS>>
    
    user financial information context: {user_info}
    
    Let's think step by step to answer: 
    '''

    query = SYS_PROMPT + "What has been the highest savings month so far this year across all accounts?" + " [/INST]"

    #     query = '''
    #     I have given some values associated with each month? Can you tell which is the highest?
    #
    # January: 41910 February: 38391 March: 31754 April: 38526 May: 36632 June: 26646 July: 19951 August: 25208 September: 27874
    #     '''

    query = '''<s>[INST]You have to act as a personal finance assistant.Let's think step by step to answer user query based on provided information.

Information:
Savings delta (calculated by taking difference of balance at start of the month and end of the month):
January: 41910 rupees
February: 38391 rupees
March: 31754 rupees
April: 38526 rupees
May: 36632 rupees
June: 26646 rupees
July: 19951 rupees
August: 25208 rupees
September: 27874 rupees
    
user query: Let's think step by step and tell what has been the highest savings month so far this year across all accounts?[/INST]
    '''
    print(query)
    print(llm.predict(query))
