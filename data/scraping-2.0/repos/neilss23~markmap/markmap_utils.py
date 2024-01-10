import os
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError
from langchain.schema.messages import HumanMessage, SystemMessage
from prompt import sparse_gpt,markdown_creator
from streamlit_markmap import markmap

load_dotenv()

def get_file_extension(file_path):
    """
    Get the extension of a file.

    Parameters:
    - file_path: Path to the file.

    Returns:
    - Extension of the file.
    """
    return os.path.splitext(file_path)[1]

def load_document_content(file):
    """
    Process files based on their extensions and print their content.

    Parameters:
    - files: List of file names.
    - filepath: Path to the directory containing the files.

    Returns:
    - None
    """

    # Define a dictionary mapping extensions to their loader classes
    loaders = {
        ".Pdf": PyPDFLoader,
        ".PDF": PyPDFLoader,
        ".pdf": PyPDFLoader,
        ".txt": TextLoader
    }

    file_extension = get_file_extension(file)
    loader_class = loaders.get(file_extension)

    if loader_class:
        loader_instance = loader_class(file)

        # Use a method based on the loader, e.g., 'load' or 'load_and_split'
        return loader_instance.load() 
    # if file_extension != ".pdf" else loader_instance.load_and_split()
    else:
        print(f"Currently unsupported file type: {file_extension}")

def read_pdf_content(item_path: str) -> str:
    # TODO: Add support for other file types (Now compatible with docx and txt)
    """
    Reads the content of a PDF file.
    """
    doc = load_document_content(item_path)
    doc_content = ""
    for page in doc:
        doc_content += page.page_content
    
    doc_content = add_space(doc_content)
    return doc_content

def add_space(s):
    lines = s.split('\n')
    processed_lines = []
    for line in lines:
        if len(line) > 0 and line[0] != ' ':
            processed_lines.append(' ' + line)
        else:
            processed_lines.append(line)
    return '\n'.join(processed_lines)

def segment_text(text, chunk_size=500, overlap=100):
    '''
    This function segments text into smaller chunks based on a character limit.
    It returns a dictionary of segmented texts.
    '''
    # Initialize the CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = chunk_size,
        chunk_overlap  = overlap,
        length_function = len,
        is_separator_regex = False,
    )
    
    # Create the documents
    documents = text_splitter.create_documents([text])
    
    # Convert to the dictionary format
    segmented_texts = {}
    segment_number = 1  # Initialize a counter for segment numbering
    
    for doc in documents:
        segmented_texts[f"Segment {segment_number}"] = doc.page_content  
        segment_number += 1 

    return segmented_texts

@retry(wait=wait_fixed(10), stop=stop_after_attempt(3))
def interact_with_model(text: str, template_string: str, model: str):
    """
    Interacts with the ChatOpenAI model and returns the output.
    
    Parameters:
    - segment (str): The text segment to process.
    - template_string (str): The template string for the prompt.
    - format_instructions (str): Formatting instructions for the prompt.
    - model (str): The name of the model to use.
    
    Returns:
    - output (dict): The model's output, or None if an error occurs.
    """
    llm = ChatOpenAI(temperature=0.0, model=model, max_retries=5, request_timeout=60)
    messages = [SystemMessage(content=template_string), HumanMessage(content=text),]
    output = llm(messages)
    return output.content

def append_to_file(text: str, filename: str = "summary.txt"):
    """
    Appends the given text to a file.

    Parameters:
    - text (str): Text to append.
    - filename (str): Name of the file to append to.
    """
    with open(filename, "a") as file:
        file.write(text + "\n")

def create_markmap(document, model="gpt-4-1106-preview"):
    """
    Processes a list of text segments and appends the results to a file.
    """
    pdf = read_pdf_content(document)
    seg = segment_text(pdf, chunk_size= 40000)
    with open("summary.txt", "w") as f:
        f.write("")
    for segment in seg:
        output = interact_with_model(segment, sparse_gpt, model)
        append_to_file(output)
    
    with open("summary.txt", "r") as f:
        summary = f.read()
    markmap = interact_with_model(summary, markdown_creator, model)

    #save markmap to markmap.md
    with open("markmap.md", "w") as f:
        f.write(markmap)

    return markmap(markmap, height=400)