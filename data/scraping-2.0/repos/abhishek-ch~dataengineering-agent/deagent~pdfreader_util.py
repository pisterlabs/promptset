from PyPDF2 import PdfReader
import tiktoken
from typing import List
from pathlib import Path
import concurrent
import openai
from deagent.utils import GPT_MODEL
from tqdm import tqdm

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    """
    Asynchronously split the provided text into n-sized chunks, ensuring that each chunk preferably
    ends at the end of a sentence or newline.

    This function attempts to tokenize and chunk the provided text in such a way that the chunks,
    where possible, end at the conclusion of a sentence or at a newline. It aims to create chunks
    of size between 0.5 * n and 1.5 * n, prioritizing chunk boundaries at the end of sentences.

    Parameters:
    - text (str): The text that needs to be split into chunks.
    - n (int): The preferred number of tokens in each chunk.
    - tokenizer: The tokenizer object used to encode and decode the text.

    Yields:
    - list of int: The list of tokens that form a chunk. Each yielded list represents a chunk of the text.

    Example:
    ```python
    tokenizer = tiktoken.get_encoding("cl100k_base")
    async for chunk in create_chunks("Sample text. Another sentence.", 10, tokenizer):
        print(chunk)
    ```

    Notes:
    - The function tries to end chunks at full stops or newlines for better coherence.
    - If no suitable end of sentence is found within the range, the chunk is created with the preferred n token size.
    """

    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


def read_pdf(filepath) -> str:
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text


def process_pdf(pdf_name: str) -> List[str]:
    """
    Process a PDF file, split its content into tokenized chunks, and prepare for summarization.

    This function reads the content of a given PDF file, tokenizes it using a specified encoding,
    and then splits the tokenized content into chunks of a defined size. Each chunk is then decoded
    back into text, which is ready for summarization.

    Parameters:
    - pdf_name (str): The PDF to be processed.

    Outputs:
    - Prints the message "Summarizing each chunk of text" once the PDF has been processed and chunks have been created.

    Returns:
    - text_chunks (list of str): List of text chunks extracted and decoded from the PDF.

    Notes:
    - Depends on the external `read_pdf`, `tiktoken.get_encoding`, and `create_chunks` functions.
    - The chosen tokenizer and chunk size is "cl100k_base" and 1500 tokens respectively.

    Example:
    ```python
    chunks = process_pdf("sample.pdf")
    ```

    """
    # This will give the directory of the current script
    current_directory = Path(__file__).parent.parent
    pdf_file_path = f"{current_directory}/pdfs/{pdf_name}"
    if not pdf_name.endswith(".pdf"):
        pdf_file_path += f".pdf"
    print(f"Reading PDF from {pdf_file_path}")
    pdf_text = read_pdf(pdf_file_path)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Chunk up the document into 1500 token chunks
    chunks = create_chunks(pdf_text, 1500, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    print(f"Summarizing each chunk of text : {len(text_chunks)}")
    return text_chunks


def extract_chunk(content, template_prompt):
    """This function applies a prompt to some input content. In this case it returns a summarized chunk of text"""
    prompt = template_prompt + content
    response = openai.ChatCompletion.create(
        model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response["choices"][0]["message"]["content"]

def summarize_chunk(text_chunks: List[str]) -> str:
    """
    Summarizes a list of text chunks concurrently using thread pooling.

    This function takes in a list of text chunks and a summary prompt. It then uses
    a ThreadPoolExecutor to concurrently process the summaries of each chunk. Progress
    is shown using a tqdm progress bar. The function finally aggregates and returns
    the summarized results.

    Parameters:
    - text_chunks (list): A list of text chunks to be summarized.
    - summary_prompt (str): A prompt used to guide the extraction or summarization process.

    Returns:
    - list: A list of summarized results for each chunk.

    Note:
    The function assumes the existence of an `extract_chunk` function which takes
    in a chunk and a summary prompt and returns the summarized or extracted data.

    """
    results = ""
    # A prompt to dictate how the recursive summarizations should approach the input paper
    summary_prompt = """Summarize this text from an academic paper. Extract any key points with reasoning.\n\nContent:"""

    # Parallel process the summaries
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(text_chunks)
    ) as executor:
        futures = [
            executor.submit(extract_chunk, chunk, summary_prompt)
            for chunk in text_chunks
        ]
        with tqdm(total=len(text_chunks)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            data = future.result()
            results += data
    return results
