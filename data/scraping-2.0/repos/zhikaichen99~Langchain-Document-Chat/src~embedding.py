from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

def embedder(model_type):
    """
    Function that returns the embedding model that will be used to embed the text

    Inputs:
        model_type - Hugging Face or OpenAI
    Outputs:
        embeddings - embedding model
    """
    if model_type == "Open AI":
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")

    return embeddings
