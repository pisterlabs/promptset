from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
import json
from langchain.chat_models import ChatOpenAI
from langchain.llms import Cohere, OpenAIChat, SagemakerEndpoint, HuggingFaceEndpoint
from langchain.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import toml
from src.models.HuggingFaceEmbeddings import InferenceEndpointHuggingFaceEmbeddings

## TO-DO replace from langchain.chat_models import ChatOpenAI

# Dirs
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR, "data/")
STATIC_DIR = Path(BASE_DIR, "static/")
# BACKGROUNDS_DIR = Path(STATIC_DIR, "background.png")
# LOGO_DIR = Path(STATIC_DIR, "logo.png")
LOGS_DIR = Path(BASE_DIR, "logs")
TOML_DIR = os.path.join(BASE_DIR, "client_config.toml")

# MODELS AND STORAGES
HF_EMBEDDING_MODEL_NAME = os.environ.get("HF_EMBEDDING_MODEL_NAME")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT", "us-west4-gcp-free")
OPENAI_API_KEY = os.environ.get("OPEN_AI_KEY")
GLOB = os.environ.get("GLOB", None)
BUCKET_NAME = os.environ.get("BUCKET_NAME")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_MODEL_NAME = os.environ.get("COHERE_EMBEDDING_MODEL_NAME")
COHERE_EMBEDDING_MODEL_NAME = os.environ.get("COHERE_EMBEDDING_MODEL_NAME")

HF_EMBEDDING_ENDPOINT = os.environ.get("HF_EMBEDDING_ENDPOINT")
HF_EMBEDDING_API_KEY = os.environ.get("HF_EMBEDDING_API_KEY")
HF_EMBEDDING_ENDPOINT_QA = os.environ.get("HF_EMBEDDING_ENDPOINT_QA")
HF_FALCON_ENDPOINT = os.environ.get("HF_FALCON_ENDPOINT")
endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
region = os.environ.get("AWS_REGION")


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": {**model_kwargs}})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]


# MODEL CATALOG
AVAILABLE_LLMS = {
    "GPT 3.5 turbo": ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.0),
    "Cohere LLM": Cohere(cohere_api_key=COHERE_API_KEY, temperature=0.0, truncate="START"),
    # "Falcon 7b": HuggingFaceEndpoint(
    #     endpoint_url=HF_FALCON_ENDPOINT, task="text-generation", huggingfacehub_api_token=HF_EMBEDDING_API_KEY
    # )
    # SagemakerEndpoint(
    # endpoint_name=endpoint_name,
    # region_name=region,
    # model_kwargs={"max_new_tokens": 500, "top_p": 0.9, "max_length": None, "temperature": 1e-10},
    # endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    # content_handler=ContentHandler(),
    # credentials_profile_name="fundamentl-ai",
    # ),
}

AVAILABLE_EMBEDDINGS = {
    "Cohere": CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model=COHERE_EMBEDDING_MODEL_NAME),
    # "stsb-xlm-r-multilingual": HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL_NAME),
    # "self_hosted_stsb-xlm-r-multilingual": InferenceEndpointHuggingFaceEmbeddings(
    #     HF_EMBEDDING_ENDPOINT, HF_EMBEDDING_API_KEY
    # ),
    # "self_hosted_multi_qa": InferenceEndpointHuggingFaceEmbeddings(HF_EMBEDDING_ENDPOINT_QA, HF_EMBEDDING_API_KEY),
}

client_config = toml.load(TOML_DIR)
TITLE = client_config["branding"]["title"]
BACKGROUNDS_DIR = client_config["branding"]["background_image_url"]
LOGO_DIR = client_config["branding"]["logo_url"]
CLIENT_DATASOURCE = client_config["available_datasources"]["client_datasource"]
CLIENT_DATASOURCE_URI = client_config["available_datasources"]["client_datasource_uri"]
HUGGING_FACE_EMBEDDINGS_ENDPOINT = client_config["Embedding_Models"]["hugging_face_endpoint"]
HUGGING_FACE_API_TOKEN = client_config["Embedding_Models"]["hugging_face_api_token"]

if __name__ == "__main__":
    print(client_config)
