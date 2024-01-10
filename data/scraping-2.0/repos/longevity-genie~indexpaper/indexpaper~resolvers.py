import os
from enum import Enum

from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings, VertexAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from loguru import logger
from pycomfort.files import *
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.huggingface import *
from indexpaper.splitting import OpenAISplitter, SourceTextSplitter, HuggingFaceSplitter

class EmbeddingModels(Enum):
    all_mpnet_base: str = "sentence-transformers/all-mpnet-base-v2"
    bge_large_en: str = "BAAI/bge-large-en" #so far best at https://huggingface.co/spaces/mteb/leaderboard
    bge_base_en_1_5: str = "BAAI/bge-base-en-v1.5" #so far second best at https://huggingface.co/spaces/mteb/leaderboard
    bge_large_en_1_5: str = "BAAI/bge-large-en-v1.5" #so far best at https://huggingface.co/spaces/mteb/leaderboard
    bge_base_en: str = "BAAI/bge-base-en" #so far second best at https://huggingface.co/spaces/mteb/leaderboard

    gte_large: str = "thenlper/gte-large"
    gte_base: str = "thenlper/gte-base"
    multilingual_e5_large: str = "intfloat/multilingual-e5-large" #supports many languages and pretty good
    biobert: str = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
    pubmedbert: str = 'pritamdeka/S-PubMedBert-MS-MARCO'
    biolord: str = 'FremyCompany/BioLORD-STAMB2-v1' #based on all-mpnet-base-v2 finetined for bio domain
    bioelectra: str = 'menadsa/S-BioELECTRA'
    biolinkbert: str = "michiyasunaga/BioLinkBERT-large" #best on https://microsoft.github.io/BLURB/leaderboard.html
    specter: str = "allenai/specter2_base"
    ada2: str = ""
    default: str = bge_base_en_1_5


class Device(Enum):
    cpu = "cpu"
    cuda = "cuda"
    ipu = "ipu"
    xpu = "xpu"
    mkldnn = "mkldnn"
    opengl = "opengl"
    opencl = "opencl"
    ideep = "ideep"
    hip = "hip"
    ve = "ve"
    fpga = "fpga"
    ort = "ort"
    xla = "xla"
    lazy = "lazy"
    vulkan = "vulkan"
    mps = "mps"
    meta = "meta"
    hpu = "hpu"
    mtia = "mtia"
    privateuseone = "privateuseone"
DEVICES =  [d.value for d in Device]


class VectorDatabase(Enum):
    Chroma = "Chroma"
    Qdrant = "Qdrant"


class EmbeddingType(Enum):
    OpenAI = "openai"
    Llama = "llama"
    VertexAI = "vertexai"
    HuggingFace = "huggingface"
    HuggingFaceBGE = "huggingface_bge"


EMBEDDINGS: list[str] = [e.value for e in EmbeddingType]
VECTOR_DATABASES: list[str] = [db.value for db in VectorDatabase]


def resolve_splitter(embeddings_type: EmbeddingType,

                     model: Optional[Union[Path, str]] = None,
                     chunk_size: Optional[int] = None
                     ) -> SourceTextSplitter:
    """
    initializes a splitter based on embedding type and additional parameters
    :param embeddings_type:
    :param model:
    :param chunk_size:
    :return:
    """
    if embeddings_type == EmbeddingType.OpenAI:
        if chunk_size is None:
            chunk_size = 3600
        return OpenAISplitter(tokens=chunk_size)
    elif embeddings_type==embeddings_type.HuggingFace or embeddings_type==embeddings_type.HuggingFaceBGE:
        if chunk_size is None:
            chunk_size = 512
        if model is None:
            logger.error("Model should be specified for Huggingface splitter, using default sentence-transformers/all-mpnet-base-v2 otherwise")
            return HuggingFaceSplitter("sentence-transformers/all-mpnet-base-v2", tokens=chunk_size)
        else:
            return HuggingFaceSplitter(model, tokens=chunk_size)
    else:
        logger.warning(f"{embeddings_type} splitter is not supported, using openai tiktoken based splitter instead")
        return OpenAISplitter(tokens=chunk_size)


def resolve_embeddings(embeddings_type: EmbeddingType, model: Optional[Union[Path, str]] = None, device: Device = Device.cpu, normalize_embeddings: bool = True) -> Embeddings:
    """
    Initializes embedding instance based on embedding type enum and on additional parameter like model and device
    :param embeddings_type:
    :param model:
    :param device:
    :return:
    """
    if embeddings_type == EmbeddingType.OpenAI:
        return OpenAIEmbeddings()
    elif embeddings_type == EmbeddingType.Llama:
        if model is None:
            model = os.getenv("LLAMA_MODEL")
            if model is None:
                logger.error(f"for llama embeddings for {model} model")
            else:
                return LlamaCppEmbeddings(model_path = model)
        return LlamaCppEmbeddings(model_path = str(model))
    elif embeddings_type == EmbeddingType.VertexAI:
        return VertexAIEmbeddings()
    elif embeddings_type == EmbeddingType.HuggingFace:
        encode_kwargs = {'normalize_embeddings': normalize_embeddings}
        if model is None:
            logger.warning(f"for huggingface the model name should be specified")
            return HuggingFaceEmbeddings(model_kwargs={'device': device.value}, encode_kwargs=encode_kwargs)
        else:
            return HuggingFaceEmbeddings(model_name = str(model), model_kwargs={'device': device.value}, encode_kwargs=encode_kwargs)
    elif embeddings_type == EmbeddingType.HuggingFaceBGE:
        model_name = "BAAI/bge-large-en" if model is None else model
        model_kwargs = {'device': device.value}
        encode_kwargs = {'normalize_embeddings': normalize_embeddings} # set True to compute cosine similarity
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        logger.warning(f"{embeddings_type.value} is not yet supported by CLI, using default openai embeddings instead")
        return OpenAIEmbeddings()

def resolve_embedding_splitter(embeddings_type: EmbeddingType, model: Optional[Union[Path, str]] = None, device: Device = Device.cpu, chunk_size=512):
    return resolve_embeddings(embeddings_type, model, device), resolve_splitter(embeddings_type, model, chunk_size=chunk_size)
