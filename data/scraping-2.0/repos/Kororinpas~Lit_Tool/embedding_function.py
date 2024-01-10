def get_embedding_function():
    from langchain.embeddings import HuggingFaceEmbeddings
    import torch 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device':device}

    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)