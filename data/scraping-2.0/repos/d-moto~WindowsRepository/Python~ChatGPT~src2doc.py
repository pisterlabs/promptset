import inspect
import os
import argparse
import faiss
import mauve
import re


from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

##################### Configuration #####################
SYSTEM_MESSAGE_TEMPLATE_FILE = "./template_system.txt"
HUMAN_MESSAGE_TEMPLATE_FILE = "./template_human.txt"
LOG_FILE = "./qa.txt"


##################### Function ####################

def parse_argument() -> argparse.Namespace:
    # """
    # Parses arguments passed to this tool.
    #
    # Args: None
    #
    # Returns:
    #     argparse.Namespace: argument store
    # """
    """
    Name: parse_argument

    Function signature: None

    File path: src2doc.py

    Key functionality and specification:
    - Parses arguments passed to this tool.

    Input: None

    Output:
    - argparse.Namespace: argument store

    Process step by step:
    1. Parses arguments passed to this tool using argparse.
    2. Returns the argument store.
    """

    parser = argparse.ArgumentParser(description='From sources to readable docs')
    parser.add_argument('-d', '--dirname', type=str, help='drepository directory')
    parser.add_argument('-f', '--filename', type=str, help='source file')
    parser.add_argument('-q', '--question', type=str, help='question file')

    args = parser.parse_args()

    return args


def is_valid_argument(arg: argparse.Namespace) -> bool:
    """
    Determines whether the specified argument pair is appropriate for use.
    Otherwise, it prints an error message to stdout.

    Args:
        arg: parsed arguments

    Peturns:
        True: appropriate for use
        False: inappropriate for use
    """

    if arg.question is None:
        print('Error: no question file was specified by -q option')
        return False

    if arg.dirname is None and arg.filename is None:
        print('Error: no directory was specified nor file path')
        return False

    if arg.dirname is not None and arg.filename is not None:
        print('Error: Cannot specify both directory and file path')
        return False

    if arg.question is None:
        print('Error: no question file')
        return False

    return True


def is_load_target(filepath: str) -> bool:
    """
    Determines whether the contents of the specified file should be included in the vector store.
    For example, git's internal files are considered excluded.

    Args:
        filepath: determined target

    Returns:
        True: it should be included
        False: it should be excluded
    """

    if ".git" in filepath:
        return False
    return True


def load_file(filepath: str, docs: list):
    """
    Adds the specified file to the document store.

    Args:
        filepath: target file
        docs: document store

    Returns: None
    """

    if not is_load_target(filepath):
        print(f"Warn: skip loading {filepath}")
        return

    try:
        loader = TextLoader(filepath, encoding='utf-8')
        docs.extend(loader.load_and_split())
    except Exception as e:
        print(f"Warn: failed to load file {filepath}: {e}")


def load_directory(dirpath: str, docs: list):
    """
    Adds the whole content in the specified directory to the document store.

    Args:
        dirpath: target directory
        docs: document store

    Returns: None
    """

    for direpath, _dirnames, filenames in os.walk(dirpath):
        for file in filenames:
            filepath = os.path.join(dirpath, file)
            load_file(filepath, docs)


def load_questions(filepath: str) -> list:
    """
    Loads questions from the given file.

    Args:
        filepath: target file

    Returns:
        list: question list splitted line by llne
    """

    with open(filepath) as f:
        return f.readlines()


def load_prompt_template() -> ChatPromptTemplate:
    """
    Constructs a prompt template for LangChain.
    System prompt from SYSTEM_MESSAGE_TEMPLATE_FILE.
    User prompt from HUMAN_MESSAGE_TEMPLATE_FILE.

    Args: None

    Returns:
        ChatPromptTemplate: a prompt template containing a system and an user prompt
    """

    with open(SYSTEM_MESSAGE_TEMPLATE_FILE) as f:
        system_template_content = f.read()
    pass
    system_template = SystemMessagePromptTemplate.from_template(system_template_content)

    with open(HUMAN_MESSAGE_TEMPLATE_FILE) as f:
        human_template_content = f.read()
    pass
    human_template = SystemMessagePromptTemplate.from_template(human_template_content)

    return ChatPromptTemplate.from_messages([system_template, human_template])


def create_vectorstore() -> FAISS:
    """
    Initializes a vector store.

    Args: None

    Returns:
        FAISS: a vector store
    """

    embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
    index = faiss.IndexFlatL2(embedding_size)
    embedding_fn = OpenAIEmbeddings().embed_query
    vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

    return vectorstore


def create_memory(vectorstore, docs: list) -> VectorStoreRetrieverMemory:
    """
    Adds source codes to given vector store, then constructs a memory for LLM.

    Args:
        vectorstore: empty vector store
        docs: loaded source codes

    Returns:
        VectorStoreRetrieverMemory: a constructed memory from the vector store
    """

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    search_kwargs = {
        'distance_metric': 'cos',
        'fetch_k': 20,
        'k': 10,
        'maximal_marginal_relevance': True
    }
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    for text in texts:
        source_filename = text.metadata['source']
        inputs = {"input": source_filename}
        outputs = {"output": text.page_content}
        memory.save_context(inputs=inputs, outputs=outputs)

    return memory


def generate_mauve_value(p_text, q_text, device_id=0, max_text_length=256, verbose=True, featurize_model_name='gpt2-large'):
    """
    Compare humantext and machinetext

    Args:
        p_text: human text
        q_text: machine text
        device_id:
        max_text_length:
        verbose:
        featurize_model_name:

    Returns:
        mauve:
    """
    out = mauve.compute_mauve(
        p_text=p_text,
        q_text=q_text,
        device_id=0,
        max_text_length=256,
        verbose=True,
        featurize_model_name='gpt2-large'
    )

    return out.mauve

def generate_documents(llm, memory, prompt: ChatPromptTemplate, questions: list) -> list:
    """
    Generates documents from each given question.

    Args:
        llm: your LLM model
        memory: a memory which contains contexts of the source codes
        prompt: a prompt template for LLM
        questions: documentation targets such as function name, class name, file name

    Returns:
        list: generated documents
    """

    chat_history = []
    p_text = []
    q_text = []

    for question in questions:
        chain = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
            # verbose=False
        )

        answer = chain.run(input=question)

        # p_text = question
        # p_text = inspect.getdoc(question)  # get docstring
        # p_text = question.__doc__
        # p_text = inspect.getdoc(eval(question))

        function_name_match = re.search(r"function\s+(\w+)\(", question)
        if function_name_match:
            function_name = function_name_match.group(1)
            # p_text = eval(function_name).__doc__  # 関数のdocstringを取得 (string)
            p_text.append(eval(function_name).__doc__)  # 関数のdocstringを取得 (list)

        # q_text = answer  # string
        q_text.append(answer)  # list

        print("\n#################################")
        print("p_text : ")
        print(p_text)
        print("")
        print(len(p_text))
        print("#################################")
        print("q_text : ")
        print(q_text)
        print("")
        print(len(q_text))
        print("#################################\n")
        mauve_ans = generate_mauve_value(p_text=p_text, q_text=q_text)

        chat_history.append((question, answer, mauve_ans))

    return chat_history


################# Main Routine ################

arg = parse_argument()

if not is_valid_argument(arg):
    exit(1)

print("Process: Load your repository...")
docs = []
if arg.dirname is not None:
    load_directory(arg.dirname, docs)
elif arg.filename is not None:
    load_file(arg.filename, docs)

print("Process: Load documentation settings...")
questions = load_questions(arg.question)
prompt = load_prompt_template()

print("Process: Setting up a vector store...")
vectorstore = create_vectorstore()
memory = create_memory(vectorstore, docs)

print("Process: Setting up LLM...")
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0)  # T=0 means moderately deterministic behavior

print("Process: generating documents...")
documents = generate_documents(llm, memory, prompt, questions)

print("Process: saving documents...")
with open(LOG_FILE, "w") as f:
    for question, answer, mauve_ans in documents:
        f.write(f"Question:\n{question}\n")
        f.write(f"Answer:\n{answer}\n\n")
        f.write(f"Mauve:\n{mauve_ans}\n\n\n")
