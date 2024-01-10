from chromadb.utils import embedding_functions
import tiktoken
import chromadb
import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

persist_directory = 'local-chromadb'
encoding = tiktoken.encoding_for_model('davinci')
tokenizer = tiktoken.get_encoding(encoding.name)
client = chromadb.PersistentClient(path="local-chromadb")
default_ef = embedding_functions.DefaultEmbeddingFunction()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ['OPENAI_API_KEY'],
    model_name="text-embedding-ada-002"
)


def count_tokens(text):
    token_count = len(tokenizer.encode(text))
    return token_count


def tk_len(text):
    token = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(token)


def chunk_text(text, max_token_size):
    tokens = text.split(" ")
    token_count = 0
    chunks = []
    current_chunk = ""

    for token in tokens:
        token_count += count_tokens(token)

        if token_count <= max_token_size:
            current_chunk += token + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = token + " "
            token_count = count_tokens(token)

    if current_chunk:
        chunks.append(current_chunk.strip())
    print("chunks", len(chunks))
    return chunks


async def SaveEmbeddings(chunks, collection_name, metadata, ids):
    collection = client.get_or_create_collection(name=collection_name)
    return collection.add(documents=chunks, metadatas=metadata, ids=ids)


async def saveFiles(text, collection_name, file_title):
    print(len(text), file_title)
    docs = chunk_text(text, 1000)
    print("DODOD", docs)
    metadata = []
    ids = []
    for i, doc in enumerate(docs):
        metadata.append({"index": str(i), "text": file_title})
        ids.append(file_title + "-" + str(i))

    print('chunky', len(docs))
    await SaveEmbeddings(docs, collection_name, metadata, ids)
    print('embeddings', len(docs))

    return 'success'


def Delete_files(collection_name):
    metadata = {}

    collection_name = client.list_collections()[0].name
    collection = client.get_collection(name=collection_name)
    collection.delete()


def build_prompt(query, context):
    """
    Builds a prompt for the LLM. #

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    More information: https://platform.openai.com/docs/guides/chat/introduction

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A prompt for the LLM (List[Dict[str, str]]).
    """

    system = {
        "role": "system",
        "content": "I am going to ask you a question, which I would like you to answer"
        "based only on the provided context, and not any other information."
        "If there is not enough information in the context to answer the question,"
        'say "I am not sure", then try to make a guess.'
        "Break your answer up into nicely readable paragraphs.",
    }
    user = {
        "role": "user",
        "content": f"The question is {query}. Here is all the context you have:"
        f'{(" ").join(context)}',
    }

    return [system, user]


def get_chatGPT_response(query, context):
    """
    Queries the GPT API to get a response to the question.

    Args:
    query (str): The original query.
    context (List): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=build_prompt(query, context),
    )

    print(response)
    return response.choices[0].message.content  # type: ignore


def response(query, collection_name, conversation_history):
    print(client.list_collections(), collection_name)
   
    collection = client.get_collection(name=collection_name)

    data = collection.query(
        query_texts=[query],
        n_results=15,
        include=["documents", "metadatas"]
    )

    response = get_chatGPT_response(query, data['documents'][0])

    print("RESPONSE", response)
    return response
