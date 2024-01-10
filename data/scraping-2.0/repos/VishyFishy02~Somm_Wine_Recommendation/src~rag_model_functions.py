# Define helper functions before querying:
from IPython.display import display, Markdown
import json
import pandas as pd
import os
import openai
from pathlib import Path
import pinecone
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import cohere
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# load config file
current_dir = Path(__file__).parent
config_file_path = current_dir.parent / 'config.json'
with open(config_file_path, 'r') as file:
    config_data = json.load(file)

OPENAI_API_KEY = config_data.get("openai_api_key")
PINECONE_API_KEY = config_data.get('pinecone_api_key')
PINECONE_ENVIRONMENT = config_data.get('pinecone_environment')
COHERE_API_KEY = config_data.get("cohere_api_key")

def get_df_for_result(res):
    if 'result' in res:  # Used for RetrievalQA
        res_text = res['result']
    elif 'answer' in res:  # Used for ConversationalRetrievalChain
        res_text = res['answer']
    elif 'response' in res:  # Used for ConversationChain
        res_text = res['response']
    else:
        raise ValueError("No 'result', 'answer', or 'response' found in the provided dictionary.")
        
    # Convert to pandas dataframe
    rows = res_text.split('\n')    
    split_rows = [r.split('|') for r in rows]
    
    split_rows_clean=[]
    for r in split_rows:
        clean_row =  [c.strip() for c in r if c!='']
        split_rows_clean.append(clean_row)
    
    # Extract the header and data rows
    header = split_rows_clean[0]
    data = split_rows_clean[2:]
    
    # Create a pandas DataFrame using the extracted header and data rows
    df = pd.DataFrame(data, columns=header)
    return df

def get_source_documents(res):
    """
    Extract and return source documents from the provided dictionary.

    Parameters:
    - res (dict): The dictionary containing the source documents.

    Returns:
    - pandas.DataFrame: A DataFrame representing the source documents.
    """
    return get_dataframe_from_documents(res['source_documents'])

def get_dataframe_from_documents(top_results):
    # Define a helper function to format the results properly:
    data = []

    for doc in top_results:
        entry = {
            'id': doc.metadata.get('id', None),
            'page_content': doc.page_content,
            'country': doc.metadata.get('country', None),
            'description': doc.metadata.get('description', None),
            'designation': doc.metadata.get('designation', None),
            'price': doc.metadata.get('price', None),
            'province': doc.metadata.get('province', None),
            'region': doc.metadata.get('region', None),
            'style1': doc.metadata.get('style1', None),
            'style2': doc.metadata.get('style2', None),
            'style3': doc.metadata.get('style3', None),
            'title': doc.metadata.get('title', None),
            'variety': doc.metadata.get('variety', None),
            'winery': doc.metadata.get('winery', None)
        }
        data.append(entry)

    df = pd.DataFrame(data)
    return df

def load_embeddings_and_rag():    
    pinecone.init(
        api_key= PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )
    index_name = pinecone.list_indexes()[0]
    index = pinecone.Index(index_name)

    openai.api_key = OPENAI_API_KEY

    model_name = 'text-embedding-ada-002'

    embed_model = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    text_field = "info"
    vectorstore = Pinecone(
        index, embed_model, text_field
    )

    # initialize LLM
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo-1106', # Or use 'gpt-4-1106-preview' (or something better/newer) for better results
        temperature=0
    )

    template = """
    You are a wine recommender. Use the CONTEXT below to answer the QUESTION. Also take into account CHAT HISTORY.

    When providing wine suggestions, suggest 5 wines by default unless the user specifies a different quantity. If the user doesn't provide formatting instructions, present the response in a table format. Include columns for the title, a concise summary of the description (avoiding the full description), variety, country, region, winery, and province.

    Ensure that the description column contains summarized versions, refraining from including the entire description for each wine.

    If possible, also include an additional column that suggests food that pairs well with each wine. Only include this information if you are certain in your answer; do not add this column if you are unsure.

    If possible, try to include a variety of wines that span several countries or regions. Try to avoid having all your recommendations from the same country.

    Don't use generic titles like "Crisp, Dry Wine." Instead, use the specific titles given in the context, and keep the descriptions short.

    Never include the word "Other" in your response. Never make up information by yourself, only use the context and chat history.

    If the question asks for more options, do not include wines from your previous answer.

    If the question states that they don't like a particular kind of wine, do not include that kind of wine in your answer. For example, if the question says 'I don't like American wines,' do not include wines whose country is the US.

    Never mention that recommendations are based on the provided context. Also never mention that the wines come from a variety of regions or other obvious things.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    CHAT HISTORY:
    {chat_history}

    ANSWER:
    """


    PROMPT_WITH_HISTORY = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )

    chain_type_kwargs = {"prompt": PROMPT_WITH_HISTORY}

    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or COHERE_API_KEY
    # init client
    co = cohere.Client(COHERE_API_KEY)

    # Create a CohereRerank compressor with the specified user agent and top_n value
    compressor = CohereRerank(
        user_agent="wine",
        top_n=20  # Number of re-ranked documents to return
    )

    # Create a ContextualCompressionRetriever with the CohereRerank compressor
    # and a vectorstore retriever with specified search parameters
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(
            search_kwargs={'k': 500},  # Number of documents for initial retrieval (before reranking)
            search_type="similarity"  # Search type
        )
    )

    # Create the ConversationBufferWindowMemory
    buffer_memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key='question',
        output_key='answer',
        return_messages=True
    )

    # Create the ConversationalRetrievalChain with SelfQueryRetriever as the retriever and ConversationBufferMemory
    conversational_retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever, # Use our compression_retriever with Cohere Reranker
        memory=buffer_memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT_WITH_HISTORY}
    )

    # Create a CohereRerank compressor for wine style
    compressor_100 = CohereRerank(
        user_agent="wine",
        top_n=100  # Number of re-ranked documents to return
    )

    # Create a ContextualCompressionRetriever with the wine style compressor
    compression_retriever_100 = ContextualCompressionRetriever(
        base_compressor=compressor_100,
        base_retriever=vectorstore.as_retriever(
            search_kwargs={'k': 500},  # Number of documents for initial retrieval (before reranking)
            search_type="similarity"  
        )
    )

    return conversational_retrieval_chain, compression_retriever_100

def get_predictions(query_text):
    result = qa_wine(query_text)
    result_df = get_df_for_result(result)
    return result_df

def get_wine_styles(query_text):
    compressed_docs = style_retriever.get_relevant_documents(query_text)
    style_df = get_dataframe_from_documents(compressed_docs)
    top3_styles = style_df['style3'].value_counts().reset_index()[:3]
    # Removing the 'count' column
    top3_styles = top3_styles.drop(columns=['count'])

    # Renaming the 'style3' column to 'styles'
    top3_styles = top3_styles.rename(columns={'style3': 'Your recommended wine styles'})
    
    return top3_styles

# Initialize rag qa chain and style retriever
qa_wine, style_retriever = load_embeddings_and_rag()


