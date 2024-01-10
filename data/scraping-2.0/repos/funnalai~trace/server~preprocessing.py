from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def chunk_text(text: str) -> list[str]:
    """Given a string of text, will chunk the text and return an array of chunked text.

    Args:
        text (str): Text to chunks

    Returns:
        list[str]: Array of chunks
    """
    text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=40)
    chunks = text_splitter.split_text(text)
    return chunks


def get_chunk_embeddings(chunks: list[str]) -> list[list[float]]:
    """Given a list of chunks, generates embeddings for each chunk via OpenAI's embedding endpoint

    Args:
        chunks (list[str]): A list of chunks to calculate the embeddings of

    Returns:
        list[list[float]]: A list of embeddings (vectors of floats) for each chunk
    """
    embeddings = OpenAIEmbeddings()
    chunk_embeddings = embeddings.embed_documents(chunks)
    return chunk_embeddings


def map_elems_to_index(arr: list[str]) -> dict[str, int]:
    """Given an array ARR, will return a dictionary with elements of the array mapped
    to their index in the array

    Args:
        arr (list[str]): Input array

    Returns:
        dict[str, int]: Dictionary mapping elements to index in array
    """
    res = dict()
    for index in range(len(arr)):
        res[arr[index]] = index
    return res


def find_relevant_chunks(prompt_subset: str, chunks: list[str], k: int, openai_api_key: str) -> list[str]:
    """Given a prompt subset, a list of chunks, & chunk embeddings, returns the most relevant chunks and their
    closest chunk neighbors (within the original chunks list)

    Args:
        prompt_subset (str): Subset of the prompt
        chunks (list[str]): List of chunks
        k (int): Number of relevant chunks to analyze – we return these chunks & chunks surrounding them

    Returns:
        list[str]: List of most relevant chunks & surrounding neighbors
    """
    # Create a mapping from chunks to their index in the chunks array
    chunks_to_index = map_elems_to_index(chunks)
    if openai_api_key:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        embeddings = OpenAIEmbeddings()

    chunk_search = FAISS.from_texts(chunks, embeddings)
    chunks_orderered_by_similarity = chunk_search.similarity_search(
        prompt_subset)
    # Take the K most relevant chunks
    relevant_chunks: list[str] = [
        document.page_content for document in chunks_orderered_by_similarity[:k]]
    # Now, we want to add the chunks that are +/- 2 chunks from the relevant chunks
    # found above
    # set of (chunk, index) tuples - we use a set to avoid adding duplicate chunks
    chunk_index_pairs: set[tuple[str, int]] = set()
    for relevant_chunk in relevant_chunks:
        relevant_chunk_index = chunks_to_index[relevant_chunk]
        # Grab the chunks +/- 2 from this relevant chunk
        for index in range(max(0, relevant_chunk_index - 2), min(relevant_chunk_index + 3, len(chunks) - 1)):
            chunk_index_pairs.add((chunks[index], index))
    # Sort the chunks by index
    sorted_chunk_index_pairs = sorted(
        list(chunk_index_pairs), key=lambda x: x[1])
    # Return only the chunks (filter out the index)
    return [chunk for chunk, _ in sorted_chunk_index_pairs]


def return_relevant_document_context(page_text: str, prompt_subset: str, k: int, openai_api_key: str) -> list[str]:
    """Given a page text & prompt, will perform semantic search to return K relevant chunks of the file

    Args:
        page_text (str): Page text 
        prompt_subset (str): Prompt subset
        k (int): Number of relevant file chunks to return

    Returns:
        list[str]: List of K most relevant file chunks
    """
    chunks = chunk_text(page_text)
    return find_relevant_chunks(prompt_subset, chunks, k, openai_api_key)


def create_document_vectorstore(page_text: str, openai_api_key: str) -> FAISS:
    chunks = chunk_text(page_text)
    print("key: ", openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_texts(chunks, embeddings)
    print(db.docstore)
    # TODO: cache and load from cache
    return db