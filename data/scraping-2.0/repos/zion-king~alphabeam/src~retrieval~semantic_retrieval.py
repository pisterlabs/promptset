import io
import os
import time
import chromadb
import tempfile
import google.generativeai as genai
from llama_index.llms import OpenAI, Gemini
from llama_index.memory import ChatMemoryBuffer
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, PromptHelper, LLMPredictor, load_index_from_storage
from llama_index.embeddings import OpenAIEmbedding, GeminiEmbedding
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.google.generativeai import GoogleVectorStore, set_google_config, genai_extension as genaix
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer, LLMRerank, CohereRerank, LongContextReorder
from llama_index import download_loader
from llama_index.schema import Document
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from google.oauth2 import service_account
from assistants.functions import *
from embedding_utils import *
import warnings
warnings.filterwarnings("ignore")

GOOGLE_API_KEY = 'AIzaSyB4Aew8oVjBgPMZlskdhdmQs27DuyNBDAY'
os.environ["GOOGLE_API_KEY"]  = GOOGLE_API_KEY

genai.configure(
    api_key=GOOGLE_API_KEY,
    client_options={"api_endpoint": "generativelanguage.googleapis.com"},
)

CHROMADB_HOST = "localhost"
COHERE_RERANK_KEY = 'p8K3ASZaficAE1YlOh9dAY3x5Tkxa8sOmCRtJOtP'


def get_index_from_vector_db(index_name):
    
    try:
        # initialize client
        db = chromadb.HttpClient(host=CHROMADB_HOST, port=8000)
    except Exception as e:
        print('<<< get_index_from_vector_db() >>> Could not connect to database!\n', e)
        return None, None
    
    # get collection and embedding size
    try:
        chroma_collection = db.get_collection(index_name)
        doc_size = chroma_collection.count()
        print('Computing knowledge base size...', doc_size)
    except Exception as e:
        print(e)
        return None, None

    start_time = time.time()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    context_window=32000 if doc_size>300 else 16000
    embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)
    llm = Gemini(api_key=GOOGLE_API_KEY, model='models/gemini-pro', temperature=0)
    print_msg = "Using Gemini Pro..."
    print(print_msg)

    service_context = ServiceContext.from_defaults(
        llm=llm,
        context_window=context_window, 
        embed_model=embed_model,
        chunk_size=1024,
        chunk_overlap=20
    )

    print('Retrieving knowledge base index from ChromaDB...')
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, 
        storage_context=storage_context,
        service_context=service_context
    )

    print(f'Index retrieved from ChromaDB in {time.time() - start_time} seconds.')
    return index, doc_size

def postprocessor_args(doc_size):
    if doc_size<30:
        return None
    
    print('Optimising context information...')
    
    # fastest postprocessor
    cohere_rerank = CohereRerank(api_key=COHERE_RERANK_KEY, top_n=30)

    # slower postprocessor
    embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model, chunk_size=256, chunk_overlap=20) # use llama_index default MockLLM (faster)

    rank_postprocessor = LLMRerank(
        choice_batch_size=10, top_n=100,
        service_context=service_context,
        parse_choice_select_answer_fn=parse_choice_select_answer_fn
    ) \
        if doc_size>100 \
            else None
    
    # node postprocessors run in the specified order
    node_postprocessors = [
        rank_postprocessor,
        cohere_rerank,
    ] \
        if doc_size>100 \
            else [cohere_rerank]

    return node_postprocessors


def parse_choice_select_answer_fn(
    answer: str, num_choices: int, raise_error: bool = False
):
    """Default parse choice select answer function."""
    answer_lines = answer.split("\n")
    # print(answer_lines)
    answer_nums = []
    answer_relevances = []
    for answer_line in answer_lines:
        line_tokens = answer_line.split(",")
        if len(line_tokens) != 2:
            if not raise_error:
                continue
            else:
                raise ValueError(
                    f"Invalid answer line: {answer_line}. "
                    "Answer line must be of the form: "
                    "answer_num: <int>, answer_relevance: <float>"
                )
        if len(line_tokens[0].split(":"))>1 and line_tokens[0].split(":")[1].strip().isdigit():
            answer_num = int(line_tokens[0].split(":")[1].strip())
            if answer_num > num_choices:
                continue
            answer_nums.append(answer_num)
            answer_relevances.append(float(line_tokens[1].split(":")[1].strip()))
    # print(answer_nums)
    return answer_nums, answer_relevances


def get_formatted_sources(response, length=100, trim_text=True) -> str:
    """Get formatted sources text."""
    from llama_index.utils import truncate_text
    texts = []
    for source_node in response.source_nodes:
        fmt_text_chunk = source_node.node.get_content()
        if trim_text:
            fmt_text_chunk = truncate_text(fmt_text_chunk, length)
        # node_id = source_node.node.node_id or "None"
        node_id = source_node.node.metadata['page_label'] or "None"
        source_text = f"> Source (Page no: {node_id}): {fmt_text_chunk}"
        texts.append(source_text)
    return "\n\n".join(texts)


def semantic_prompt_style(): 

    prompt_header = f"""Your name is Alpha, a highly intelligent system for conversational business intelligence.
    Your task is to use the provided knowledege base, containing semantic models of a business dataset,
    to determine if my question can be answered based on the semantic knowledge. 
    If the semantic knowledge contains the parameter(s) useful for answering my question, respond with a Yes, and No otherwise.
    """

    return prompt_header   


def query_gen_prompt_style(): 

    with open("./assistants/mf_few_shot.txt", "r") as f:
        # print(file)
        few_shot_examples = f.read()

    prompt_header = f"""Your name is Alpha, a highly intelligent system for conversational business intelligence. 
    As an SQL and dbt MetricFlow expert, your goal is to use the provided knowledege base, containing dbt metrics and 
    semantic models of a business dataset, to generate a MetricFlow query command needed to answer the question.
    The provided knowledge base also contains the MetricFlow documentation which provides a comprehensive description of the MetricFlow command syntax.
       
    To generate the correct MetricFlow query command, determine the following:
    1. Which metrics are needed to answer the question? --metrics
    2. Which tables and dimensions contain the required data? --group-by <table_name__dimension_name>
    4. Is there a specific time interval requested? --start-time 'YYYY-MM-DD' --end-time 'YYYY-MM-DD' 
    5. Do the results need to be filtered by a specific condition? --where
    6. Do the results need to be ordered by a specific table dimension? --order
    7. Is there a limit on the number of records requested? --limit
    
    Here's a few examples of how a business requirement is translated into a MetricFlow query command in accordance with the semantic models in the knowledge base.
    {few_shot_examples}

    Following these examples, generate a single-line query command that answers the question, using the relevant metric, table and dimension names obtained from the dbt semantic and metric models.
    If a time interval is requested, dont include --where in the command, use --start-time and --end-time instead
    Use a double underscore to link every dimension name to its corresponding table name <table__dimension>, as in the following examples:
       - order date is in the sales_key table hence it must be referenced as `sales_key__order_date`
       - product category is in the product table hence it must be referenced as `product__product_category`
       - In your --group-by, where, or --order expressions, don't ever use only dimension names without linking with the table name.
    Return only this command without any further explanation, or return "Null" if the provided information is not sufficient to answer the question.
    """
    # The measures, dimensions, tables and other parameters referenced in the above examples are 
    # obtained from the knowledege base provided below. 
     
    # The general syntax for the MetricFlow query command is:
    # `mf query --metrics <measure(s)> --group-by <table-1__dimension-1,table-1__dimension-2,...,table-n__dimension-n> --where <condition> --start-time 'YYYY-MM-DD' --end-time 'YYYY-MM-DD' --order <table__dimension> --limit <int>`

    # 3. Which dimensions in the tables contain the required data?

    return prompt_header   

def retrieved_data_prompt_style(llm_query_input):

    prompt_header = f"""Your name is Alpha, a highly intelligent system for 
    conversational business intelligence. Your task is to provide data analysis and interpretation 
    to a non-technical audience using the provided data, which has been fetched from a database.
    If the provided data doesn't contain the answer to the question, return the data it contains, with a comprehensive explanation.                
    
    The data was retrieved as a table and converted to text. The Metricflow command used to generate the provided data is as follows:
    {llm_query_input}

    This should give you additional context about how the database was queried to fetch the data.
    The retrieved data contains the relevant metrics and dimensions useful for answering the question.
    Ensure that you interpret the data carefully whether or not it answers the question, and let your response be as comprehensive and explanatory as possible. 
    Never decline to answer my question. If you are unable to answer the question, interpret the available information in the data and return a contextual explanation of what the data contains.
    """

    return prompt_header   


def answer_query_stream(query, index_name, prompt_style):

    index, doc_size = get_index_from_vector_db(index_name)

    if index is None:
        response = "Requested information not found"
        return response
    else:
        node_postprocessors = postprocessor_args(doc_size)
        similarity_top_k = 200 if doc_size>200 else doc_size
        chat_engine = index.as_chat_engine(chat_mode="context", 
                                            # memory=chat_history, # shouldn't retain chat history here
                                            system_prompt=prompt_style, 
                                            similarity_top_k=similarity_top_k,
                                            function_call="query_engine_tool",
                                            node_postprocessors=node_postprocessors
                                            )

        message_body = f"""\nUse the tool to answer:\n{query}\n"""
        response = chat_engine.chat(message_body)
        
        if response is None:
            chat_response = "I'm sorry I couldn't find an answer to the requested information in your semantic knowledge base. Please rephrase your question and try again."
            return chat_response
        else:
            print('Starting response stream...\n...........................\n...........................')
            return f'''{response.response}'''


def fetch_data(user_query, llm_query_input, chat_history):
    
    with tempfile.TemporaryDirectory() as temp_dir:
        query_output_dir = llm_run_query_cmd(llm_query_input, temp_dir)

        if query_output_dir is None:
            return "Could not fetch data from the database. Please try again and if the problem persists, inform your IT team."
        
        doc = SimpleDirectoryReader(input_dir=query_output_dir).load_data()
    # doc = SimpleDirectoryReader(input_dir="./retrieval/data").load_data()

    service_context = ServiceContext.from_defaults(
        llm=Gemini(model='models/gemini-pro', temperature=0),
        embed_model=GeminiEmbedding(),
        chunk_size=1024,
        chunk_overlap=20
        )
    
    index = VectorStoreIndex.from_documents(doc, service_context=service_context)

    chat_engine = index.as_chat_engine(chat_mode="context", 
                                        memory=chat_history,
                                        system_prompt=retrieved_data_prompt_style(llm_query_input), 
                                        similarity_top_k=30,
                                        verbose=True, 
                                        # streaming=True,
                                        function_call="query_engine_tool",
                                        )

    message_body = f"""\n{user_query}\n"""
    response = chat_engine.chat(message_body)
    
    if response is None:
        print("Index retrieved but couldn't stream response...")
        chat_response = "I'm sorry I couldn't find an answer to the requested information in your database. Please rephrase your question and try again."
        return chat_response
    else:
        print('Starting response stream...\n...........................\n...........................')
        return response.response

def init_chat_history():
    new_conversation_state = ChatMemoryBuffer.from_defaults(token_limit=50000)
    return new_conversation_state













