from langchain.text_splitter import CharacterTextSplitter

def get_chunks(documents):
    # if new_files:
        
    char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator=' ')
    chunks = char_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        if len(chunk.page_content)>1000:
            chunks[i].page_content = chunk.page_content[:1000]
    return chunks