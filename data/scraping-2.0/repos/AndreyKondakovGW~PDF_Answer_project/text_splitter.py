from langchain.text_splitter import CharacterTextSplitter

def split_text(text):
    text_spliteer = CharacterTextSplitter(separator='\n',
                                          chunk_size=500, chunk_overlap=100,
                                          length_function=len)
    chunks = text_spliteer.split_text(text)
    return chunks