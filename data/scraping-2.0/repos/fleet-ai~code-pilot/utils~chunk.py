from langchain.text_splitter import NLTKTextSplitter


def chunk_nltk(text):
    nltk_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=250)
    return nltk_splitter.split_text(text)
