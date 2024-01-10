import PyPDF2, tempfile, tiktoken

from langchain.document_loaders import UnstructuredFileLoader, TextLoader

def get_pages_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        pages = []
        num_pages = len(reader.pages)

        for i in range(num_pages):
            page_text = reader.pages[i].extract_text()
            

            if i > 0:
                prev_page_text = reader.pages[i - 1].extract_text()
                page_text = prev_page_text[-300:] + page_text

            if i < num_pages - 1:
                next_page_text = reader.pages[i + 1].extract_text()
                page_text = page_text + next_page_text[:300]
                
            pages.append(page_text)
        return pages
    
def create_temp_files_from_strings(strings):
    temp_file_paths = []
    for string in strings:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(string.encode())
            temp_file_paths.append(tmpfile.name)
    return temp_file_paths

def compute_number_of_tokens(string: str, encoding_name: str = "gpt2") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def parse_into_pages(paths):
    documents = []

    for i, path in enumerate(paths):
        loader = TextLoader(path)
        document = loader.load()
        document[0].metadata = {'page': str(i+1)}  # Les pages commencent généralement à 1, pas à 0
        documents.append(document[0])
    return documents