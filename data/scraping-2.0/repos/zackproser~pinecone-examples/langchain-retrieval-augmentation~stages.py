from uuid import uuid4

import pinecone
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from text_processing import split_text_records
from tqdm.auto import tqdm


def init_stage(
        index_name: str,
        pinecone_api_key: str,
        pinecone_environment: str
):
    """
    Initialize the pinecone index, if it doesn't already exist
    """
    print(f"Stage init running...")
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment
    )

    print(f"Checking if desired pinecone index {index_name} exists...")
    if index_name not in pinecone.list_indexes():
        # Create the index
        print(f"Creating index {index_name}...")
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )
        print(f"Index {index_name} created")

    else:
        print(f"Index {index_name} already exists")


def upsert_stage(
        index_name: str,
        pinecone_api_key: str,
        pinecone_environment: str,
        openai_api_key: str
):
    """
    Pre-process the example dataset
    Upsert the data into the desired index
    """

    BATCH_LIMIT = 100
    texts = []
    metadatas = []

    print("Stage upsert running...")

    # Use OpenAI's text embedding ada-002 model
    MODEL_NAME = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=MODEL_NAME,
        openai_api_key=openai_api_key
    )

    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment
    )

    index = pinecone.GRPCIndex(index_name)

    print("Loading dataset...")
    data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')

    for i, record in enumerate(tqdm(data)):
        # first get the metadata fields for this record
        metadata = {
            'wiki-id': str(record['id']),
            'source': record['url'],
            'title': record['title']
        }
        # Now, we create chunks from the record text
        record_texts = split_text_records(record['text'])
        # Create individual metadata dicts for each chunk
        record_metadatas = [{
            'chunk': j,
            'text': text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # If we have reached the batch_limit we can add texts
        if len(texts) >= BATCH_LIMIT:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))

    index_stats = index.describe_index_stats()
    print(
        f"Upsert routine complete...Pinecone index stats: {index_stats}")


def query_stage(
        index_name: str,
        pinecone_api_key: str,
        pinecone_environment: str,
        openai_api_key: str,
        query: str
):

    print("Stage query running...")

    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment
    )

    text_field = 'text'

    index = pinecone.Index(index_name)

    # Use OpenAI's text embedding ada-002 model
    MODEL_NAME = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=MODEL_NAME,
        openai_api_key=openai_api_key
    )

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    results = qa.run(query)

    print(f"Query results: {results}")


def teardown_stage(
        index_name: str,
        pinecone_api_key: str,
        pinecone_environment: str
):
    print("Stage teardown running...")

    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment
    )

    print(f"Deleting Pinecone index {index_name}...")
    pinecone.delete_index(index_name)
    print("Teardown complete")
