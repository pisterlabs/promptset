import os

from llama_index import VectorStoreIndex, LLMPredictor, PromptHelper, Document, \
    StringIterableReader, SlackReader, LangchainEmbedding, ServiceContext
from langchain import HuggingFaceHub, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM


# ModelConfig contains the configuration for the application
class ModelConfig:
    def __init__(self, max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit, temperature, model_name,
                 local):
        self.max_input_size = max_input_size
        self.num_outputs = num_outputs
        self.max_chunk_overlap = max_chunk_overlap
        self.chunk_size_limit = chunk_size_limit
        self.temperature = temperature
        self.model_name = model_name
        self.local = local

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ModelConfig):
            return self.chunk_size_limit == o.chunk_size_limit and self.max_chunk_overlap == o.max_chunk_overlap and self.num_outputs == o.num_outputs and self.max_input_size == o.max_input_size and self.temperature == o.temperature and self.model_name == o.model_name and self.local == o.local
        return False


# get_model_config_from_env creates a ModelConfig with defaulting from the environment variables
def get_model_config_from_env() -> ModelConfig:
    max_input_size_str = os.getenv("MAX_INPUT_SIZE") if "MAX_INPUT_SIZE" in os.environ else "2048"
    num_outputs_str = os.getenv("NUM_OUTPUTS") if "NUM_OUTPUTS" in os.environ else "5096"
    max_chunk_overlap_str = os.getenv("MAX_CHUNK_OVERLAP") if "MAX_CHUNK_OVERLAP" in os.environ else "28"
    chunk_size_limit_str = os.getenv("CHUNK_SIZE_LIMIT") if "CHUNK_SIZE_LIMIT" in os.environ else "600"
    temperature_str = os.getenv("TEMPERATURE") if "TEMPERATURE" in os.environ else "0.6"
    local_str = os.getenv("LOCAL") if "LOCAL" in os.environ else "True"
    model_name = os.getenv("MODEL_NAME") if "MODEL_NAME" in os.environ else (
        "gpt-3.5-turbo" if local_str == "False" else "declare-lab/flan-alpaca-base")

    return ModelConfig(int(max_input_size_str), int(num_outputs_str), int(max_chunk_overlap_str),
                       int(chunk_size_limit_str), float(temperature_str), model_name, local_str == "True")


# get_prompt_helper creates an OpenAI PromptHelper instance
def get_prompt_helper(model_restrictions: ModelConfig) -> PromptHelper:
    return PromptHelper(model_restrictions.max_input_size, model_restrictions.num_outputs,
                        float(model_restrictions.max_chunk_overlap / model_restrictions.chunk_size_limit), chunk_size_limit=model_restrictions.chunk_size_limit)


# get_vector_index creates a GPTSimpleVectorIndex from GPTIndex Documents of any form, requires LLM and ModelConfig to be specified, also allows non OpenAI Embeddings
def get_vector_index(documents: list[Document], llm: LLM, model_config: ModelConfig,
                     embeddings=None) -> VectorStoreIndex:
    predictor = LLMPredictor(llm=llm)
    prompt_helper = get_prompt_helper(model_config)
    if embeddings is None:
        service_context = ServiceContext.from_defaults(llm_predictor=predictor, prompt_helper=prompt_helper)
        return VectorStoreIndex(documents, service_context=service_context)
    else:
        service_context = ServiceContext.from_defaults(llm_predictor=predictor, prompt_helper=prompt_helper, embed_model=embeddings)
        return VectorStoreIndex(documents, service_context=service_context)


# get_openai_api_llm constructs an OpenAI API powered LLM model, requires OPENAI_API_TOKEN to be in environment
# variables
def get_openai_api_llm(model_config):
    return ChatOpenAI(temperature=model_config.temperature, model_name=model_config.model_name,
                      max_tokens=model_config.num_outputs)


# get_local_llm_from_huggingface downlaods and constricts an LLM Model based on name from the HuggingFace repository
def get_local_llm_from_huggingface(model_config):
    return HuggingFacePipeline.from_model_id(
        model_id=model_config.model_name, task="text2text-generation",
        model_kwargs={
            "temperature": model_config.temperature,
            # "model_max_length": model_config.num_outputs,
            "max_length": model_config.num_outputs}
    )


# IndexMaker provides utility wrappers for getting indexes from either slack or plain text list using either OpenAI
# API models or a local one
class IndexMaker:

    @staticmethod
    def get_index_from_text(list_of_text: list[str]):
        documents = StringIterableReader().load_data(list_of_text)
        model_config = get_model_config_from_env()
        return get_vector_index(documents, get_openai_api_llm(model_config), model_config)

    @staticmethod
    def get_index_from_slack(channel_ids: list[str]):
        documents = SlackReader().load_data(channel_ids)
        model_config = get_model_config_from_env()
        return get_vector_index(documents, get_openai_api_llm(model_config), model_config)

    @staticmethod
    def get_hf_index_from_text(list_of_text: list[str]):
        documents = StringIterableReader().load_data(list_of_text)
        model_config = get_model_config_from_env()
        hf = IndexMaker.get_hf_embeddings()
        return get_vector_index(documents, get_local_llm_from_huggingface(model_config), model_config, hf)

    @staticmethod
    def get_hf_index_from_docs(documents: list[Document]):
        model_config = get_model_config_from_env()
        hf = IndexMaker.get_hf_embeddings()
        return get_vector_index(documents, get_local_llm_from_huggingface(model_config), model_config, hf)

    @staticmethod
    def get_hf_index_from_slack(channel_ids: list[str]):
        documents = SlackReader().load_data(channel_ids)
        model_config = get_model_config_from_env()
        hf = IndexMaker.get_hf_embeddings()
        return get_vector_index(documents, get_local_llm_from_huggingface(model_config), model_config, hf)

    @staticmethod
    def get_hf_embeddings():
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        return LangchainEmbedding(hf)

    @staticmethod
    def get_hf_llm_predictor():
        model_config = get_model_config_from_env()
        model = get_local_llm_from_huggingface(model_config)
        return LLMPredictor(llm=model)
