# Importing required libraries
import os
import shutil

import requests
import base64
from typing import Tuple
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from web_scraper import get_repository_links
from tqdm import tqdm

# Initializing global and environment variables
VECTOR_DB_PATH = "./VectorStore"

with open("tokens.txt", "r") as git_token_file:
    tokens = git_token_file.readlines()
    for token in tokens:
        token = token.rstrip('\n')
        app_name = token.split(',')[0]
        token_val = token.split(',')[-1]
        if 'github' in app_name:
            os.environ['GITHUB_TOKEN'] = token_val
        elif 'openai' in app_name:
            os.environ['OPENAI_API_KEY'] = token_val
        elif 'activeloop' in app_name:
            os.environ['ACTIVELOOP_TOKEN'] = token_val
        else:
            pass
print("Github Token:", os.environ.get('GITHUB_TOKEN'))
print("OpenAI Token:", os.environ.get('OPENAI_API_KEY'))
print("Activeloop Token:", os.environ.get('ACTIVELOOP_TOKEN'))

NON_TEXT_EXTENSIONS = [".jpg", ".jpeg", ".jpe", ".jif", ".jfif", ".jfi", ".mp3", ".mp4", ".ind", ".indt", ".indd",
                       ".mov", ".png", ".gif", ".webp", ".tiff", ".tif", ".psd", ".raw", ".bmp", ".dib", ".heif",
                       ".heic", ".eps", ".ai", ".svg", ".mkv", ".avi", ".pyc", ".pt", ".pth", ".pb", ".h5", ".ckpt",
                       ".rar", ".zip", ".tar", ".iso", ".dat", ".ico", ".docx", ".pptx", ".xlsx", ".xls", ".ppt",
                       ".doc", ".gitignore", ".nii"]
PROGRAMMING_LANGUAGES = {".py": "python", ".cpp": "C++", ".c": "C", ".h": "C or C++ header", ".java": "Java",
                         ".r": "R", ".html": "Hyper Text Markup Language (HTML)", ".css": "Cascading Style Sheet",
                         ".js": "JavaScript", ".ipynb": "Python Jupyter Notebook", ".md": "Markdown", ".php": "PHP",
                         ".cs": "C Sharp", ".ts": "TypeScript", ".sql": "SQL", ".tex": "Latex"}
QA_dict = dict()
QA_CHAIN_TYPE = "refine"
QA_MAX_TOKEN_LIMIT = 4090
MODEL_TEMPERATURE = 0.4


def parse_github_url(url: str) -> Tuple[str, str]:
    """
    The parse_github_url(url) function is designed to extract the owner and repository name from a given GitHub
    repository URL.
    Parameters
    ----------
    url : str
       github url.
    Returns
    ----------
    owner : str
        Owner name of repository.
    repo : str
        Repository name.
    """
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo


def get_files_from_github_repo(owner: str, repo: str):
    """
    The get_files_from_github_repo function collects the files from a given url.
    Parameters
    ----------
    owner : str
        Owner name of repository.
    repo : str
        Repository name.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    elif response.status_code == 404:
        # Might be the default main branch is called 'main' and not 'master', hence changing the url
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.json()
            return content["tree"]
        else:
            print("in main and response.status_code != 200", url)
            raise ValueError(f"Error fetching repo contents: {response.status_code}")
    else:
        print("in master and response.status_code != 200 and !=404", url)
        raise ValueError(f"Error fetching repo contents: {response.status_code}")


def fetch_contents(files: list) -> list:
    """
    The fetch_contents function fetches contents from the given list of files.
    Parameters
    ----------
    files : list
        List of files.
    Returns
    -------
    contents: list[Documents]
        Returns list of documents.
    """
    contents = []
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github+json"
    }

    def is_non_text_file(filename: str, extensions: list) -> bool:
        """
        The is_non_text_file function checks if the file is text or non text.
        Parameters
        ----------
        filename : str
            List of files.
        extensions : str
            File extensions.
        Returns
        -------
            Returns True if files extensions matches with the param 'extension',
            else return False.
        """
        return any(filename.endswith(e) for e in extensions)

    def get_programming_language(filename: str) -> str:
        """
        The get_programming_language function checks in which programming language the filename is written.
        Parameters
        ----------
        filename : str
            filename.
        Returns
        -------
            Returns the programming language of the filename,
            else returns "Unknown".
        """
        ext = filename.split('.')[-1]
        lang = PROGRAMMING_LANGUAGES.get("." + ext)
        if lang is None:
            return "Unknown"
        else:
            return lang

    # Extracting contents from file
    for file in tqdm(files):
        if file["type"] == "blob" and not is_non_text_file(file["path"], NON_TEXT_EXTENSIONS):
            response = requests.get(file["url"], headers=headers)
            if response.status_code == 200:
                content = response.json()["content"]
                try:
                    decoded_content = base64.b64decode(content).decode('utf-8')
                    repository_name = file['url'].split('/')[5]
                    contents.append(Document(page_content=decoded_content, metadata={"source": file['path'],
                                                                                     "repository": repository_name,
                                                                                     "programming_language:": get_programming_language(
                                                                                         file['path'])}))
                except Exception as e:
                    print(f"Decoding error {e}")
                    print("For file: ", file['path'])
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return contents


def get_chunks_from_files(files, chunk_size=1024, chunk_overlap=20):
    """
    The get_chunks_from_files function gets chunks of data from files.
    Parameters
    ----------
    files : str
        Link of repository.
    chunk_size : int
        Default value set to 1024.
    chunk_overlap : int
        Default value set to 20.
    Returns
    -------
    source_chunks: list
        List of chunks of documents extracted from the files.
    """
    source_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = fetch_contents(files)
    total_files = str(len(documents))
    for source in documents:
        meta_data = source.metadata
        meta_data['total number of files'] = total_files
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=meta_data))
    return source_chunks


def create_questions_db(qa_dict: dict):
    """
    This function collates all the questions and answers collected from asking questions about each repository to the
    GPT. Then it creates a new vector store database to store these collated questions and answers with all required
    formatting like Splitting and chunking. Then it returns the database's retriever object.
    :return: retriever: Database's retirever object.
    """
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(dataset_path=os.path.join(VECTOR_DB_PATH, "final"), embedding_function=embeddings)
    question_answer_string = ""
    for repo, chats in qa_dict.items():
        question_answer_string += "The following questions and answers are for repository named " + repo + "\n"
        for (question, answer) in chats:
            question_answer_string += "Question: " + question + "\n"
            question_answer_string += "Answer: " + answer + "\n"
    print(question_answer_string)
    data_chunks = list()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    for chunk in splitter.split_text(question_answer_string):
        data_chunks.append(Document(page_content=chunk))
    db.add_documents(data_chunks)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    # retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20
    retriever.search_kwargs["reduce_k_below_max_tokens"] = True

    return retriever


def create_repo_db(repo_url: str, embedding_func):
    """
    This function parses the files from the GitHub repository, creates chunks and fetches them into the vector store
    database. Then it returns the database's retriever object.
    :param repo_url: The URL of the GitHub repository.
    :param embedding_func: The embedding Function required for Vector Store database
    :return: retriever: the retriever object that will be used when creating question_answer chain object.
    """
    print("\nGetting files from ", repo_url)
    github_username, repo_name = parse_github_url(repo_url)
    if os.path.exists(os.path.join(VECTOR_DB_PATH, repo_name)):
        print("Removing already existing vector store and creating new one")
        shutil.rmtree(os.path.join(VECTOR_DB_PATH, repo_name))
    db = DeepLake(dataset_path=os.path.join(VECTOR_DB_PATH, repo_name), embedding_function=embedding_func)
    files = get_files_from_github_repo(github_username, repo_name)
    print("Splitting the files into chunks")
    data_chunks = get_chunks_from_files(files)
    print("Adding extracted chunks to Vector Store at", VECTOR_DB_PATH)
    db.add_documents(data_chunks)

    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20
    retriever.search_kwargs["reduce_k_below_max_tokens"] = True
    return retriever


def ask_gpt(db_retriever, questions, auto_questions=True):
    """
    This function creates a Conversational chain that uses the provided vector store database and the created GPT model
    to prompt the model with questions supplied.
    :param db_retriever: Retreiver object of the vector store Database
    :param questions: List of questions to ask the GPT model
    :param auto_questions: If False, the user can ask questions himself/herself. (Used for prompt testing)
    :return: chat_history: The collection of questions and answers as list of question,answer tuples.
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=MODEL_TEMPERATURE)  # 'ada' 'gpt-3.5-turbo' 'gpt-4',
    qa = ConversationalRetrievalChain.from_llm(model, retriever=db_retriever, max_tokens_limit=QA_MAX_TOKEN_LIMIT,
                                               chain_type=QA_CHAIN_TYPE)

    chat_history = []

    def ask_question(question):
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(f"**Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")

    if not auto_questions:
        while True:
            question = input("Enter Question: ")
            ask_question(question)
            choice = int(input("Press 0 to quit: "))
            if choice == 0:
                break
    else:
        for question in questions:
            ask_question(question)
    return chat_history


# main function
if __name__ == "__main__":
    # TEST Repos
    repo_links = [
        "https://github.com/arijitde92/Alzheimer_Classification",
        # "https://github.com/ishitade123/D",
        "https://github.com/arijitde92/AD_Classification_App",
        "https://github.com/hwchase17/chroma-langchain",
        "https://github.com/ISHITA1234/flasklogin",
        # "https://github.com/arijitde92/Phantom_Generator"
    ]
    embeddings = OpenAIEmbeddings(disallowed_special=())

    #############################################
    #         REPOSITORY LEVEL PROCESSING       #
    #############################################
    for repo_url in tqdm(repo_links):
        _, repo_name = parse_github_url(repo_url)
        repo_retriever = create_repo_db(repo_url, embeddings)
        repo_questions = [
            # "Give a breif description of this repository from the code snippets provided in your context",
            "How difficult or complex are the code snippets in your context, for an entry level junior developer?"
        ]
        QA_dict[repo_name] = ask_gpt(repo_retriever, repo_questions, False)

    #############################################
    #         PROFILE LEVEL PROCESSING          #
    #############################################
    final_retirever = create_questions_db(QA_dict)
    final_questions = [
        # "Which repositories are we discussing about?",
        # "Tell me a brief summary of each repository",
        "Compare the complexities of the repositories in the context and say which is the most technically complex "
        "for an entry level junior developer fresh out of college?"
    ]
    final_answers = ask_gpt(final_retirever, final_questions, False)
    print("Conclusion:")
    print(final_answers[-1][-1])
