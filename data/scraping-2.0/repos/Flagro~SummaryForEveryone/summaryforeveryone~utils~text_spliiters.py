from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import NLTKTextSplitter


def get_text_splitter(separator="\n\n", chunk_size=1024, text_splitter_name="character"):
    if text_splitter_name == "spacy":
        return SpacyTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=0,  
                separator=separator, 
                pipeline="sentencizer"
                )
    elif text_splitter_name == "character":
        return CharacterTextSplitter.from_tiktoken_encoder(
                separator=separator,
                chunk_size=chunk_size // 10,
                chunk_overlap=0,
            )
    elif text_splitter_name == "nltk":
        return NLTKTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=0,  
                separator=separator
            )
    else:
        raise ValueError(f"Unknown text splitter name: {text_splitter_name}")
