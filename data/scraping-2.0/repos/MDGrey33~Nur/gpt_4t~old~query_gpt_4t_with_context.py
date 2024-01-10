import os
import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LLM
from gpt_4t_response.load_document import process_document
from credentials import oai_api_key
from configuration import file_system_path
from configuration import vector_chunk_folder_path

# Set API key in environment variables
os.environ["OPENAI_API_KEY"] = oai_api_key

# Create OpenAI client and wrapper class
client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])


class OpenAIWrapper(LLM):
    def __init__(self, client):
        self.client = client

    def generate(self, prompt, **kwargs):
        # Adjust this method to match LangChain's expectations
        return self.client.Completion.create(prompt=prompt, **kwargs)

# Instantiate the OpenAI wrapper
openai_wrapper = OpenAIWrapper(client)


def get_all_files_in_directory(directory):
    """
    Returns a list of all file paths within the specified directory and its subdirectories.

    Args:
    - directory (str): The path to the directory.

    Returns:
    - list: A list of file paths.
    """
    file_paths = []

    # Walk through directory and subdirectories
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.endswith('.DS_Store'):
                file_paths.append(os.path.join(dirpath, filename))

    return file_paths


def initial_load_context():
    """
    Initially load context into the vector database.
    """
    # Get a list of all files in the 'context/' directory and subdirectories
    file_paths = get_all_files_in_directory(file_system_path)

    # Process each file using the process_document function
    all_documents = []
    for file_path in file_paths:
        documents = process_document(file_path)
        all_documents.extend(documents)  # Combine all documents into a single list
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    # create the vector database from the context folder
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=vector_chunk_folder_path)
    vectordb.persist()


def load_context_and_answer(question, debug=True):
    vectordb = Chroma(persist_directory=vector_chunk_folder_path, embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(llm=openai_wrapper, chain_type="stuff", retriever=retriever, return_source_documents=True)

    llm_response = qa_chain(question)
    answer = llm_response['result']

    if debug:
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])

    return answer

def load_chat():
    debug = True
    while True:
        print("You: (type 'done' on a new line when finished)")
        user_input = ""
        while True:
            line = input()
            if line.lower() == 'done':
                break
            user_input += line + '\n'
        if user_input.lower() in ['exit\n', 'quit\n']:
            break
        print('Processing...')
        answer = load_context_and_answer(user_input, debug)
        print(f'GPT: {answer}')


if __name__ == "__main__":
    initial_load_context()
    load_chat()

