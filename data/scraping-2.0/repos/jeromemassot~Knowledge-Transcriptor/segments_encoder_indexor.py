from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
import pandas as pd
import openai
import json
import re
import os


def encode_and_index(folder_path, collection_name:str, chosen_metadata, qdrant_api_key:str) -> str:
    """
    Encode the segments in the folder and index them into a vector database.
    :param folder_path: path to the folder containing the segments
    :param collection_name: name of the collection to create
    :param chosen_metadata: list of metadata to enrich the vectors
    :param qdrant_api_key: API key for the Qdrant vector database
    :return: message with the number of segments indexed
    """

    # retrieve the additional information stored in a payload
    csv_file_name = os.path.basename(folder_path) + ".csv"
    metadata_df = pd.read_csv(os.path.join("./inputs", csv_file_name), sep='\t')
    metadata_df = metadata_df[chosen_metadata]

    # clean the title in order to create an unique identifier
    metadata_df['Id'] = metadata_df['Title'].apply(lambda x: re.sub('[^\w]', '', x))
    metadata_df['Id'] = metadata_df['Id'].apply(lambda x: abs(hash(x)) % (10 ** 8))
    metadata_dict = metadata_df.set_index('Id').T.to_dict("dict")

    # load the encoder
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # load the segments and the payloads and create the indexes
    all_segments = []
    payloads = []
    idx = []

    for segment_file in os.listdir(folder_path):
        segment_path = os.path.join(folder_path, segment_file)
        id = re.sub('[^\w]', '', segment_file[:-6].replace("_", ""))
        id = abs(hash(id)) % (10 ** 8)
        payload = metadata_dict[id]
        with open(segment_path, "r") as f:
            segments = [json.loads(l) for l in f.readlines()]
            for i, segment in enumerate(segments):
                all_segments.append(segment)
                current_segment_payload = payload.copy()
                current_segment_payload.update(
                    {
                        "Text": segment["text"],
                        "Start": segment["timestamp"][0], 
                        "End": segment["timestamp"][1]}
                    )
                payloads.append(current_segment_payload)
                idx.append(abs(hash(id+i)) % (10 ** 10))
    
    # create the index
    qdrant_client = QdrantClient(
        url="https://08afe25e-6838-46ed-946e-f36b8d2afe10.eu-central-1-0.aws.cloud.qdrant.io:6333", 
        api_key=qdrant_api_key
    )

    # create the collection
    qdrant_client.recreate_collection(
	    collection_name=collection_name,
	    vectors_config=models.VectorParams(
		    size=encoder.get_sentence_embedding_dimension(),
		    distance=models.Distance.COSINE
	    )
    )

    # upload the segments
    qdrant_client.upload_records(
	    collection_name=collection_name,
	    records=[
		    models.Record(
			    id=id,
			    vector=encoder.encode(segment["text"]).tolist(),
			    payload=payload
		    ) for id, segment, payload in zip(idx, all_segments, payloads)
	    ]
    )

    return f"Segments indexed: {len(all_segments)}"


def query_index(question:str, collection_name:str, qdrant_api_key:str, top_k:int=3) -> list:
    """
    Query the index and return the results.
    :param question: question to query
    :param top_k: number of contexts to return
    :return: list of results
    """
    # create the index
    qdrant_client = QdrantClient(
        url="https://08afe25e-6838-46ed-946e-f36b8d2afe10.eu-central-1-0.aws.cloud.qdrant.io:6333", 
        api_key=qdrant_api_key
    )

    # load the encoder
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    hits = qdrant_client.search(
	    collection_name=collection_name,
	    query_vector=encoder.encode(question).tolist(),
	    limit=top_k
    )
    
    return hits


def list_collections(qdrant_api_key:str) -> list:
    """
    List the collections in the index.
    :param qdrant_api_key: API key for the Qdrant vector database
    :return: list of collections
    """

    # create the index
    qdrant_client = QdrantClient(
        url="https://08afe25e-6838-46ed-946e-f36b8d2afe10.eu-central-1-0.aws.cloud.qdrant.io:6333", 
        api_key=qdrant_api_key
    )

    collection_names = [collection[1][0].name for collection in qdrant_client.get_collections()]
    return collection_names


def answer_question(question:str, contexts:list, openai_api_key:str) -> str:
    """
    Answer the question with the contexts.
    :param question: question to answer
    :param contexts: list of contexts
    :param openai_api_key: API key for the OpenAI API
    :return: answer
    """

    openai.api_key = openai_api_key

    context = " ".join(contexts)

    prompt = f"""Answer the question based on the context below and if the question can't be answered based on the context, say \'I don't know\'\n\n
    Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
    """

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            model="text-davinci-003"
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)

