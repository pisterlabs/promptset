from langchain.document_loaders import TextLoader, PythonLoader, BSHTMLLoader, PDFPlumberLoader


def load_text_file(file_path):
    loader = TextLoader(file_path)
    documents = loader.load_and_split()
    return documents


def load_python_file(file_path):
    loader = PythonLoader(file_path)
    documents = loader.load_and_split()
    return documents


def load_bshtml_file(file_path):
    loader = BSHTMLLoader(file_path)
    documents = loader.load_and_split()
    return documents


def load_pdf_file(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load_and_split()
    return documents


def process_document(file_path):
    # Determine the file extension
    file_extension = file_path.split('.')[-1]

    # Call the appropriate function based on file extension
    if file_extension == 'txt':
        return load_text_file(file_path)
    elif file_extension == 'py':
        return load_python_file(file_path)
    elif file_extension == 'html' or file_extension == 'htm':
        return load_bshtml_file(file_path)
    elif file_extension == 'pdf':
        return load_pdf_file(file_path)
    else:
        print(f"Warning: Unsupported file extension: {file_extension} for file {file_path}. Skipping...")
        return []
