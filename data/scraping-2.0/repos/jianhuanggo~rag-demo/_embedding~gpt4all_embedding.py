from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from _common import _common as _common_


@_common_.exception_handler
def gpt4all_embedding() -> HuggingFaceEmbeddings:
    """
    Creates an instance of HuggingFaceEmbeddings with a specified transformer model.

    This function initializes an embedding model specifically designed for paraphrase
    detection using the 'sentence-transformers/paraphrase-MiniLM-L6-v2' model. It
    automatically detects the available device (GPU or CPU) for running the model.

    Returns:
        HuggingFaceEmbeddings: An instance of the HuggingFaceEmbeddings class configured
        with the paraphrase detection transformer model.

    """
    device = f"cuda: {cuda.current_device()}" if cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": 32},
    )
