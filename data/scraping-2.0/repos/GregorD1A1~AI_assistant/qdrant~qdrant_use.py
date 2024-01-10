from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Distance, VectorParams, PointStruct
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.embeddings import OpenAIEmbeddings
from InstructorEmbedding import INSTRUCTOR
from airtable import Airtable
import os
import sys
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

qdrant = QdrantClient("localhost", port=30217)

memory_collection = "Memory"
tools_collection = "tools"


airtable_token = os.getenv('AIRTABLE_API_TOKEN')
friends_table = Airtable('appGWWQkZT6s8XWoj', 'tblRws3jW42T7BteV', airtable_token)
services_table = Airtable('appGWWQkZT6s8XWoj', 'tblACGeBESjN8GUeX', airtable_token)
tech_knowledge_table = Airtable('appGWWQkZT6s8XWoj', 'tbl4nxFlrurlCFgrE', airtable_token)
types = {'friend': friends_table, 'service': services_table, 'tech_knowledge': tech_knowledge_table}

embeddings_openai = OpenAIEmbeddings()
#embeddings_opensorce = INSTRUCTOR('hkunlp/instructor-large')


def create_collection_and_upsert(collection, type):
    is_indexed = next(
        (collection for collection in qdrant.get_collections().collections if collection.name == collection), None
    )
    # Create empty collection if not exists
    if not is_indexed:
        qdrant.create_collection(
            collection,
            #vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            on_disk_payload=True
        )

    collection_info = qdrant.get_collection(collection)

    # Add data to collection if empty
    if not collection_info.points_count:
        upsert_data(collection, type)


def upsert_data(collection, type):
    for table_type in types:
        print(table_type)
        rows = types[table_type].get_all()
        print(rows)
        points = []
        # Generate embeddings and index data
        for row in rows:
            payload = row['fields']
            payload['type'] = type
            #embedding = embeddings_opensource.encode(payload['content'])
            embedding = embeddings_openai.embed_query(payload['content'])
            points.append({
                'id': payload['uuid'],
                'payload': payload,
                'vector': embedding
            })
        print(points)
        qdrant.upsert(
            collection_name=collection,
            wait=True,
            points=[
                PointStruct(id=point['id'], vector=point['vector'], payload=point['payload']) for point in points
            ]
        )


def vector_search(query, type):
    sys.stdout.write("embedding query...")
    sys.stdout.flush()
    # for ada embedding
    query_embedding = embeddings_openai.embed_query(query)
    # for instructor embedding
    #query_embedding = embeddings_opensorce.encode(query)
    results = qdrant.search(
        collection_name=memory_collection,
        query_vector=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=type))]
        ),
        limit=3,
    )
    print(results)
    rerank_results = rerank_filter(query, results)
    sys.stdout.write(rerank_results)
    sys.stdout.flush()
    search_output = ''
    for i, result in enumerate(results):
        if int(rerank_results[i]) == 1:
            search_output += f"{result.payload['name']}: {result.payload['content']}"

    if search_output == '':
        search_output = 'Nothing found.'

    return search_output


def rerank_filter(query, results):
    batch = []
    for result in results:
        batch.append({'query': query, 'result': result.payload['content']})
    prompt = ("Check if provided record answers query:"
              "\n###\nQuery:\n'''{query}'''\n\nRecord:\n'''{result}'''\n\nReturn '1' if answers or '0' if not. "
              "You are strictly forbidden to return anything else except '0' or '1'")
    prompt_template = PromptTemplate.from_template(prompt)
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    chain = prompt_template | llm | StrOutputParser()
    rerank_result = chain.batch(batch)

    return rerank_result


if __name__ == '__main__':
    upsert_data(memory_collection, 'service')
    #print(vector_search('App for fine-tuning models in cloud', 'service'))
