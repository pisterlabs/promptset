from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n", "."],
        chunk_size = 800,
        chunk_overlap = 150,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks