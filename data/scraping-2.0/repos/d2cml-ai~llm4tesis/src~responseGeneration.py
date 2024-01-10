from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from prompts import *
from dotenv import load_dotenv
import pinecone
import tiktoken

load_dotenv()

pinecone.init()
embedding = OpenAIEmbeddings()
vector_database = Pinecone.from_existing_index(
    index_name="llm4tesis", 
    embedding=embedding
)
retriever = vector_database.as_retriever(search_type="mmr")

def query_handler(query):
    relevant_docs = retriever.get_relevant_documents(query)
    context = get_page_contents(relevant_docs)
    query_with_context = human_template.format(query=query, context=context)
    return {"role": "user", "content": query_with_context}

def message_token_count(message, num_tokens, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    for key, value in message.items():
        num_tokens += len(encoding.encode(value))

        if key == "name": num_tokens -= 1
        
    return num_tokens
    
def num_tokens_from_messages(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    if model != "gpt-4":
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. 
                                  See https://github.com/openai/openai-python/blob/main/chatml.md 
                                  for information on how messages are converted to tokens.""")
    
    num_tokens = 0

    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n        
        num_tokens = message_token_count(message, num_tokens, model)
    
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def ensure_fit_tokens(messages, max_tokens = 4096):
    """
    Ensure that total tokens in messages is less than MAX_TOKENS.
    If not, remove oldest messages until it fits.
    """
    total_tokens = num_tokens_from_messages(messages)

    while total_tokens > max_tokens:
        messages.pop(0)
        total_tokens = num_tokens_from_messages(messages)
    
    return messages

def get_page_contents(docs):
    contents = ""

    for i, doc in enumerate(docs, 1):
        docMetadata = doc.metadata
        page = doc.page_content
        title = docMetadata["title"]
        author = docMetadata["author"]
        advisor = docMetadata ["advisor"]
        year = docMetadata["year"]
        url = docMetadata["url_thesis"]
        contents += f"Document #{i}:\n"
        contents += f"Title: {title}\n"
        contents += f"Author(s): {author}\n"
        contents += f"Advisor: {advisor}\n"
        contents += f"Year: {year}\n"
        contents += f"Link: {url}\n"
        contents += page + "\n\n"
    
    return contents

def construct_messages(history):
    messages = [{"role": "system", "content": system_prompt}]

    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role": role, "content": entry["message"]})
    
    messages = ensure_fit_tokens(messages)
    return messages