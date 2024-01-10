from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain

from llama_index import GPTVectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext, SimpleDirectoryReader
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.keyword_table import GPTSimpleKeywordTableIndex


load_dotenv()

def create_index(filename):

    loader = SimpleDirectoryReader("./data/"+filename, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    print("Loaded {} documents".format(len(documents)))
    print("First document: {}".format(documents[0]))

    index = None
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm_predictor = LLMPredictor(llm)
    max_input_size = 4096
    num_output = 1024
    max_chunk_overlap = 64
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    print("Creating index")
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    print("Index created")
    print("Index conv_id: {}".format(index.index_id))
    print(index.docstore)

    if not os.path.exists("./persist/"+filename):
        os.makedirs("./persist/"+filename)
    index.storage_context.persist(persist_dir="./persist/"+filename)
    print("./persist/"+filename)


def load_multiple_indexes():
    # Define the directories
    directories = ['./persist/1', './persist/2']

    # Create a dictionary of StorageContext objects
    storage_contexts = {}
    for directory in directories:
        storage_contexts[directory] = StorageContext.from_defaults(persist_dir=directory)

    # Load the indexes from each directory
    ind = []
    vector_indices = {}
    for directory, storage_context in storage_contexts.items():
        index = load_index_from_storage(storage_context)
        ind.append(index)
        vector_indices[directory] = index

    index_summaries = ["This document is about how humans have evolved to become the dominant species on Earth, "
                       "and how our current way of life is causing a rapid rate of extinction of other species. It "
                       "proposes that humans should be treated as a collective organism, and examines the past, "
                       "present, and future of our species in order to diagnose and treat the ailments caused by our "
                       "current way of life.","This document is about the philosophical debate on the idea of truth, "
                                              "the importance of understanding how identity is built up through a "
                                              "network of interactions, and the prevalence of individualism in "
                                              "American culture."]

    # Create the composed graph
    graph = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex,
        [ind[0], ind[1]],
        index_summaries=index_summaries,
        max_keywords_per_chunk=100
    )

    # Create the query engine
    query_engine = graph.as_query_engine()
    # query_engine = ind[1].as_query_engine()

    query = "What is cancer culture?"

    _DEFAULT_TEMPLATE = """Given the context information answer the following question
                                    If you don't know the answer, just say you dont know Don't try to make up an answer.
                                    =========
                                    Always answer in the format:
                                    ANSWER: <your answer>
                                    FOLLOW UP QUESTIONS: <list of 3 suggested questions related to context and conversation for better understanding>
                                    SOURCE: <do not make up source, give the page or the chapter from the document>
                                    =========
                                    question: {}""".format(query)

    response = query_engine.query(query)
    print(response)
    print(response.get_formatted_sources())


