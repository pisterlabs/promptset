from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
import chromadb
from helpers import pinecone_connect


def compare_lpas(topic, lpa_1='Tamworth_Borough_Council', lpa_2='Northumberland_County_Council'):

    index = pinecone_connect()

    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")

    # embed query
    query = embedding_model.embed_documents(
        [topic],
    )

    # Query
    lpa_1_results = index.query(
        vector=query,
        filter={
            "LPA": {"$eq": lpa_1},
        },
        top_k=8,
        include_metadata=True
    )

    lpa_1_results_string = ''
    for match in lpa_1_results['matches'][0:8]:

        lpa_1_results_string += match['metadata']['text'] + '\n'

    lpa_2_results = index.query(
        vector=query,
        filter={
            "LPA": {"$eq": lpa_2},
        },
        top_k=8,
        include_metadata=True
    )

    lpa_2_results_string = ''
    for match in lpa_2_results['matches'][0:8]:

        lpa_2_results_string += match['metadata']['text'] + '\n'

    context = '''
    Context from {lpa_1}: 
    {lpa_1_results}

    Context from {lpa_2}: 
    {lpa_2_results}

    '''.format(lpa_1_results=lpa_1_results_string,
               lpa_2_results=lpa_2_results_string,
               lpa_1=lpa_1,
               lpa_2=lpa_2
               )

    print('context length')
    print(len(context))

    print(context)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1.5)

    query_string = '''
    Compare the way the following 2  approach to the topic of {topic}, listing the three biggest differences. 

    {context} 
    '''. format(context=context, topic=topic)

    result = llm.call_as_llm(query_string)

    return result
