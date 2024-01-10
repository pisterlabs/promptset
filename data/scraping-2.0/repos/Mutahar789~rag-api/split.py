from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
token_based_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=250, chunk_overlap=10
)

recursive_character_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 50,
    length_function = len,
)

class csv_splitter():
    def transform_documents(docs):
        return docs

splitter_dict = {
    ".pdf": token_based_splitter,
    ".txt": token_based_splitter,
    ".docx": token_based_splitter,
    ".dox": token_based_splitter,
    ".odt": token_based_splitter,
    ".wav": token_based_splitter,
    ".mp3": token_based_splitter,

    ".html": recursive_character_spliter,
    ".md": recursive_character_spliter,
    ".pptx": recursive_character_spliter,
    ".xlsx": recursive_character_spliter,
    ".xls": recursive_character_spliter,
    ".csv": csv_splitter,
}

to_remove = [Language.MARKDOWN, Language.HTML]
languages = [lang for lang in Language if lang not in to_remove]

for lang in languages:
    splitter_dict[f".{lang.value}"] = RecursiveCharacterTextSplitter.from_language(
        language=lang, chunk_size=60, chunk_overlap=0
    )

def split(docs, ext):
    splitter = splitter_dict.get(ext, recursive_character_spliter)
    split_docs = splitter.transform_documents(docs)
    return split_docs