import os
import time
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.chat_models import ChatOpenAI

# Download necessary NLTK models and data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize constants
CORPUS_FOLDER = "corpus"
MAX_FINAL_RESPONSE_SIZE = 200
MAX_CHUNK_RESPONSE_SIZE = 100

# Retrieve the set of English stopwords
stop_words = set(stopwords.words('english'))

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")

def tokenize(document):
    """
    Tokenize a document, remove stopwords and non-alphabetic tokens.

    Args:
        document (str): The document to tokenize.

    Returns:
        list: List of tokens from the document.
    """
    # Tokenize the document
    tokens = word_tokenize(document.lower())
    # Remove stopwords and non-alphabetic tokens
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

def load_documents():
    """
    Load and tokenize documents from the "corpus" folder.

    Returns:
        list: List of tokenized documents.
    """
    print("Loading and tokenizing documents...")
    tokenized_documents = []
    for filename in os.listdir(CORPUS_FOLDER):
        if filename.endswith('.txt'):  # Ensure we're reading text files
            with open(os.path.join(CORPUS_FOLDER, filename), 'r', encoding='iso-8859-1') as file:
                # Read and tokenize the document
                text = file.read()
                tokens = tokenize(text)
                tokenized_documents.append(tokens)
    print(f"Loaded and tokenized {len(tokenized_documents)} documents.")
    return tokenized_documents

def retrieve_documents_with_bm25(query, tokenized_documents, top_k, remove_top_2=False):
    """
    Retrieve top documents using the BM25 algorithm.

    Args:
        query (str): The user query.
        tokenized_documents (list): List of tokenized documents.
        top_k (int): Number of top documents to retrieve.
        remove_top_2 (bool): If true, exclude the top 2 documents from the result.

    Returns:
        list: List of top documents based on BM25 scores.
    """
    print(f"Retrieving top-{top_k} documents with BM25...")
    bm25 = BM25Okapi(tokenized_documents)
    query_tokens = tokenize(query)
    doc_scores = bm25.get_scores(query_tokens)

    # If remove_top_2 is True, adjust the slicing to exclude the top 2 documents
    if remove_top_2:
        start_index = 2
    else:
        start_index = 0

    top_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[start_index:top_k]
    top_documents = [tokenized_documents[i] for i in top_doc_indices]
    print(f"{len(top_documents)} documents retrieved.")
    return top_documents

def generate_prompts(document, user_query, max_token_limit=16385):
    """
    Generate document chunks as prompts for the LLM.

    Args:
        document (str): The document to chunk.
        user_query (str): The user query for context.
        max_token_limit (int): Maximum token limit for each chunk.

    Returns:
        list: List of generated prompts.
    """
    prompt_intro = f"The user is seeking information related to '{user_query}'. Given this context, carefully analyze the following text. Identify and summarize the key points that directly address the user's query in less than 100 tokens, especially focusing on any relevant facts, insights, or advice. Highlight critical details and provide a concise summary that would be most helpful and informative to the user:\n\n"
    
    # Estimate the number of tokens in the prompt
    estimated_prompt_tokens = len(prompt_intro) // 4

    # Calculate the maximum size for each chunk in terms of tokens
    max_words_per_chunk = (max_token_limit - estimated_prompt_tokens - MAX_CHUNK_RESPONSE_SIZE) // 2 

    # Split the document into words
    words = document.split()

    # Create document chunks based on max_words_per_chunk and convert to individual prompts
    prompt_chunks = []
    for i in range(0, len(words), int(max_words_per_chunk)):
        chunk = ' '.join(words[i:i + int(max_words_per_chunk)])
        prompt_chunks.append(prompt_intro + chunk)

    return prompt_chunks

def batch_process(prompts):
    """
    Process prompts in batches. A 5 second gap between batches is used because of rate limiting.

    Args:
        prompts (list): List of prompts to process.

    Returns:
        str: Concatenated responses from all batches.
    """
    responses = ""

    for idx, i in enumerate(range(0, len(prompts), 3)):
        print(f"Processing Batch {idx+1}/{(len(prompts)//3)+1}")
        # Extract a batch of 3 prompts
        batch = prompts[i:i+3]

        # Send the batch to the LLM
        response_batch = llm.batch(batch, max_tokens=MAX_CHUNK_RESPONSE_SIZE)

        # Extract each text response from the batch
        for response in response_batch:
            responses += response.content

        # Wait for 5 seconds before sending the next batch
        time.sleep(5)
    
    return responses

def run_system():
    """
    Main function to run the document retrieval and summarization system.
    """
    start = time.time()
    user_query = input("Please enter your mental health-related query: ")
    tokenized_documents = load_documents()
    top_k = 5 # Change here for top k documents
    remove_top_2 = False # Change here to test removing top 2 documents

    # Retrieve top k documents using BM25
    top_documents = retrieve_documents_with_bm25(user_query, tokenized_documents, top_k, remove_top_2)

    all_prompts = []

    # Generate prompts from top-K documents
    for tokenized_doc in top_documents:
        doc_string = ' '.join(tokenized_doc)
        all_prompts += generate_prompts(doc_string, user_query)
    
    # Batch process prompts
    combined_responses = batch_process(all_prompts)

    final_prompt = (
        f"Summary of Information: {combined_responses}\n\n"
        "Based on this earlier summary about cognitive-behavioral therapy (CBT) techniques, "
        f"apply these strategies directly through a normal conversational style of a counselor in 200 tokens or less to help a user's current situation and questions: {user_query}"
    )

    final_response = llm.predict(final_prompt, max_tokens=MAX_FINAL_RESPONSE_SIZE)
    print(f"Response: {final_response}")
    end = time.time()
    print(f"Response Duration: {end-start}")

if __name__ == "__main__":
    run_system()