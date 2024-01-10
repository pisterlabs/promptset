from langchain.text_splitter import CharacterTextSplitter

class TextSplitter():
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
