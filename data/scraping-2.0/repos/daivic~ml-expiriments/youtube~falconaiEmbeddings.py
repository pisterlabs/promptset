import os

from langchain import HuggingFaceHub
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import textwrap

embeddings = HuggingFaceEmbeddings()

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# --------------------------------------------------------------
# Load the LLM model from the HuggingFaceHub
# --------------------------------------------------------------

repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": .1, "max_new_tokens": 3000}
)


def create_db_from_file_path(path):
    loader = TextLoader(path)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    print("database created")
    return db

def replace_newlines_with_spaces(file_path):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace newlines with spaces
    content = content.replace('\n', ' ')

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)

    print(f"Newlines replaced with spaces in the file: {file_path}")

def get_response_from_query(db, query, k=2):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    # Template to use for the system message prompt
    template = """
        This is a cookbook: {docs}
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "answer the question given the cookbook: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=falcon_llm, prompt=chat_prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


# Example usage:
path_to_file = "../database"
replace_newlines_with_spaces(path_to_file)

db = create_db_from_file_path(path_to_file)

query = "Give me a sandwich recipe that has salmon"
response, docs = get_response_from_query(db, query)
print(docs)
print(textwrap.fill(response, width=50))
