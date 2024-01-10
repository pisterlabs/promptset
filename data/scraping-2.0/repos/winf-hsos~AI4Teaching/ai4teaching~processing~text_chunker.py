from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

class TextChunker:
    def __init__(self):
        pass

    def chunk(self, text, chunk_size=1000, chunk_overlap_pct=0.1, split_character="\n\n"):

        text_splitter = CharacterTextSplitter(
            separator = split_character,
            chunk_size = chunk_size,
            chunk_overlap  = int(chunk_size * chunk_overlap_pct),
            length_function = len,
            is_separator_regex = False,
        )

        langchain_document = Document(page_content=text)
        splits = text_splitter.split_documents([langchain_document])
        
        split_documents = [ { "text":  s.page_content, "metadata" : s.metadata } for s in splits]
        return split_documents


     