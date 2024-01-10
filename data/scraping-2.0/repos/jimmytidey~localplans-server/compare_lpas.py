from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from helpers import pinecone_connect


def compare_lpas(topic, lpa_1, lpa_2):

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
            "LPA": {"$eq": lpa_1 + '.txt'},
        },
        top_k=8,
        include_metadata=True
    )

    lpa_1_results_string = ''
    for match in lpa_1_results['matches'][0:8]:
        print('-------')
        print(match['score'])
        print(match['metadata']['text'])
        if (match['score'] > 0.3):
            lpa_1_results_string += match['metadata']['text'] + '\n'
        else:
            lpa_1_results_string += f'{lpa_1} does not have a detailed policy on {topic}' + '\n'

    lpa_2_results = index.query(
        vector=query,
        filter={
            "LPA": {"$eq": lpa_2 + '.txt'},
        },
        top_k=8,
        include_metadata=True
    )

    lpa_2_results_string = ''
    for match in lpa_2_results['matches'][0:8]:
        print('-------')
        print(match['score'])
        print(match['metadata']['text'])
        if (match['score'] > 0.3):
            lpa_2_results_string += match['metadata']['text'] + '\n'
        else:
            lpa_2_results_string += f'{lpa_2} does not have a detailed policy on {topic}' + '\n'

    context = '''
    Context from {lpa_1}: 
    {lpa_1_results}

    Context from {lpa_2}: 
    {lpa_2_results}

    '''.format(lpa_1_results=lpa_1_results_string,
               lpa_2_results=lpa_2_results_string,
               lpa_1=lpa_1.replace('_', ' '),
               lpa_2=lpa_2.replace('_', ' ')
               )

    print('context length')
    print(len(context))

    print(context)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    query_string = '''
    Compare the way the following two areas approach to the topic of {topic}, listing the differences. 

    {context} 
    '''. format(context=context, topic=topic)

    result = llm.call_as_llm(query_string)

    return result
