from langchain.text_splitter import CharacterTextSplitter

def get_chunk_text(text):
    """ 
    Get chunk text 

    Parameters:
    text (string) : all pdf text

    Returns:
    list : returns the text chunk list 

    """
    
    text_spliter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_spliter.split_text(text)
    return chunks