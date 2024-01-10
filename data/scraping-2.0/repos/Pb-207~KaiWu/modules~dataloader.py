import tiktoken
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from modules.llm import cf

# Create the length function
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )

    return len(tokens)


# Split str to chunks, input str return a list of chunks
def split_chunk(chunk):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cf.getint('PDF', 'chunk_size'),
        chunk_overlap=cf.getint('PDF', 'chunk_overlap'),
        length_function=tiktoken_len
    )

    return text_splitter.split_text(chunk)


# Load PDFs and split to chunks
def pdf_loader(pdf_path):
    loader = PyPDFDirectoryLoader(pdf_path)
    page = loader.load()
    index = []
    chunk = []
    chunks = []
    cut = -5 * cf.getint('PDF', 'chunk_overlap')
    print('Loading PDFs ...')
    for i in tqdm(range(len(page))):
        # Set overlaps for pages
        if i > 0 and page[i].metadata.get('source') == page[i - 1].metadata.get('source'):
            page[i].page_content = page[i - 1].page_content[cut:] + page[i].page_content

        # Split and filter pages to chunks
        chunk.append(split_chunk(page[i].page_content))
        for j in range(len(chunk[i])):
            index.append(page[i].metadata)
            chunks.append(chunk[i][j].replace('\n', '').replace(' -', ''))

    return chunks, index