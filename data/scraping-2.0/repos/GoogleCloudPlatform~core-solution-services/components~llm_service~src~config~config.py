# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
  LLM Service config file
"""
# pylint: disable=unspecified-encoding,line-too-long,broad-exception-caught
import os
from common.utils.logging_handler import Logger
from common.utils.secrets import get_secret
from common.utils.token_handler import UserCredentials
from schemas.error_schema import (UnauthorizedResponseModel,
                                  InternalServerErrorResponseModel,
                                  ValidationErrorResponseModel)
from google.cloud import secretmanager
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.llms.cohere import Cohere
from langchain.llms.vertexai import VertexAI
from langchain.embeddings.openai import OpenAIEmbeddings

Logger = Logger.get_logger(__file__)
secrets = secretmanager.SecretManagerServiceClient()

PORT = os.environ["PORT"] if os.environ.get("PORT") is not None else 80
PROJECT_ID = os.environ.get("PROJECT_ID")
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
GCP_PROJECT = PROJECT_ID
DATABASE_PREFIX = os.getenv("DATABASE_PREFIX", "")
REGION = os.getenv("REGION", "us-central1")

try:
  with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r",
            encoding="utf-8", errors="ignore") as ns_file:
    namespace = ns_file.readline()
    JOB_NAMESPACE = namespace
except FileNotFoundError as e:
  JOB_NAMESPACE = "default"
  Logger.info("Namespace File not found, setting job namespace as default")

CONTAINER_NAME = os.getenv("CONTAINER_NAME")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_BASE_URL = os.getenv("API_BASE_URL")
SERVICE_NAME = os.getenv("SERVICE_NAME")
SKAFFOLD_NAMESPACE = os.getenv("SKAFFOLD_NAMESPACE")
GKE_CLUSTER = os.getenv("GKE_CLUSTER")
GCP_ZONE = os.getenv("GCP_ZONE")

PAYLOAD_FILE_SIZE = 1024

ERROR_RESPONSES = {
    500: {
        "model": InternalServerErrorResponseModel
    },
    401: {
        "model": UnauthorizedResponseModel
    },
    422: {
        "model": ValidationErrorResponseModel
    }
}

# LLM configuration

def get_environ_flag(env_flag_str, default=True):
  default_str = str(default)
  evn_val = os.getenv(env_flag_str, default_str)
  if evn_val is None or evn_val == "":
    evn_val = default_str
  evn_flag = evn_val.lower() == "true"
  Logger.info(f"{env_flag_str} = {evn_flag}")
  return evn_flag


# VertexAI models are enabled by default
ENABLE_GOOGLE_LLM = get_environ_flag("ENABLE_GOOGLE_LLM", True)
ENABLE_GOOGLE_MODEL_GARDEN = get_environ_flag("ENABLE_GOOGLE_MODEL_GARDEN",
                                              True)

# 3rd party models are enabled if the flag is set AND the API key is defined
ENABLE_OPENAI_LLM = get_environ_flag("ENABLE_OPENAI_LLM", True)
ENABLE_COHERE_LLM = get_environ_flag("ENABLE_COHERE_LLM", True)

# truss hosted models
ENABLE_TRUSS_LLAMA2 = get_environ_flag("ENABLE_TRUSS_LLAMA2", True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
  OPENAI_API_KEY = get_secret("openai-api-key")
ENABLE_OPENAI_LLM = ENABLE_OPENAI_LLM and (OPENAI_API_KEY is not None)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if COHERE_API_KEY is None:
  COHERE_API_KEY = get_secret("cohere-api-key")
ENABLE_COHERE_LLM = ENABLE_COHERE_LLM and (COHERE_API_KEY is not None)

OPENAI_LLM_TYPE_GPT3_5 = "OpenAI-GPT3.5"
OPENAI_LLM_TYPE_GPT4 = "OpenAI-GPT4"
OPENAI_EMBEDDING_TYPE = "OpenAI-Embeddings"
COHERE_LLM_TYPE = "Cohere"
VERTEX_LLM_TYPE_BISON_TEXT = "VertexAI-Text"
VERTEX_LLM_TYPE_BISON_V1_CHAT = "VertexAI-Chat-V1"
VERTEX_LLM_TYPE_BISON_CHAT = "VertexAI-Chat"
VERTEX_LLM_TYPE_GECKO_EMBEDDING = "VertexAI-Embedding"
VERTEX_AI_MODEL_GARDEN_LLAMA2_CHAT = "VertexAI-ModelGarden-LLAMA2-Chat"
TRUSS_LLM_LLAMA2_CHAT = "Truss-Llama2-Chat"

LLM_TYPES = []
OPENAI_LLM_TYPES = [OPENAI_LLM_TYPE_GPT3_5, OPENAI_LLM_TYPE_GPT4]
COHERE_LLM_TYPES = [COHERE_LLM_TYPE]
GOOGLE_LLM_TYPES = [VERTEX_LLM_TYPE_BISON_TEXT,
                    VERTEX_LLM_TYPE_BISON_V1_CHAT,
                    VERTEX_LLM_TYPE_BISON_CHAT]

# these LLMs are trained as chat models
CHAT_LLM_TYPES = [OPENAI_LLM_TYPE_GPT3_5,
                  OPENAI_LLM_TYPE_GPT4,
                  VERTEX_LLM_TYPE_BISON_V1_CHAT,
                  VERTEX_LLM_TYPE_BISON_CHAT,
                  TRUSS_LLM_LLAMA2_CHAT,
                  VERTEX_AI_MODEL_GARDEN_LLAMA2_CHAT]

if ENABLE_OPENAI_LLM:
  LLM_TYPES.extend(OPENAI_LLM_TYPES)

if ENABLE_COHERE_LLM:
  LLM_TYPES.extend(COHERE_LLM_TYPES)

if ENABLE_GOOGLE_LLM:
  LLM_TYPES.extend(GOOGLE_LLM_TYPES)

LANGCHAIN_LLM = {}
if ENABLE_OPENAI_LLM:
  LANGCHAIN_LLM.update({
    OPENAI_LLM_TYPE_GPT3_5: ChatOpenAI(temperature=0,
                                       openai_api_key=OPENAI_API_KEY,
                                       model_name="gpt-3.5-turbo"),
    OPENAI_LLM_TYPE_GPT4: ChatOpenAI(temperature=0,
                                     openai_api_key=OPENAI_API_KEY,
                                     model_name="gpt-4"),
    COHERE_LLM_TYPE: Cohere(cohere_api_key=COHERE_API_KEY, max_tokens=1024)
  })

GOOGLE_LLM = {}
if ENABLE_GOOGLE_LLM:
  GOOGLE_LLM = {
    VERTEX_LLM_TYPE_BISON_TEXT: "text-bison",
    VERTEX_LLM_TYPE_BISON_V1_CHAT: "chat-bison@001",
    VERTEX_LLM_TYPE_BISON_CHAT: "chat-bison-32k",
    VERTEX_LLM_TYPE_GECKO_EMBEDDING: "textembedding-gecko@001"
  }
  LANGCHAIN_LLM.update({
    VERTEX_LLM_TYPE_BISON_TEXT: VertexAI(
      model_name=GOOGLE_LLM[VERTEX_LLM_TYPE_BISON_TEXT], project=PROJECT_ID),
    VERTEX_LLM_TYPE_BISON_V1_CHAT: ChatVertexAI(
      model_name=GOOGLE_LLM[VERTEX_LLM_TYPE_BISON_V1_CHAT], project=PROJECT_ID),
    VERTEX_LLM_TYPE_BISON_CHAT: ChatVertexAI(
      model_name=GOOGLE_LLM[VERTEX_LLM_TYPE_BISON_CHAT], project=PROJECT_ID)
  })

GOOGLE_MODEL_GARDEN = {}
if ENABLE_GOOGLE_MODEL_GARDEN:
  MODEL_GARDEN_LLAMA2_CHAT_ENDPOINT_ID = \
    os.getenv("MODEL_GARDEN_LLAMA2_CHAT_ENDPOINT_ID")
  if MODEL_GARDEN_LLAMA2_CHAT_ENDPOINT_ID:
    GOOGLE_MODEL_GARDEN_TYPES = [VERTEX_AI_MODEL_GARDEN_LLAMA2_CHAT]
    GOOGLE_MODEL_GARDEN = {
        VERTEX_AI_MODEL_GARDEN_LLAMA2_CHAT: MODEL_GARDEN_LLAMA2_CHAT_ENDPOINT_ID,
    }
    LLM_TYPES.extend(GOOGLE_MODEL_GARDEN_TYPES)

LLM_TRUSS_MODELS = {}
if ENABLE_TRUSS_LLAMA2:
  LLM_TRUSS_MODEL_ENDPOINT = os.getenv("TRUSS_LLAMA2_ENDPOINT")
  if LLM_TRUSS_MODEL_ENDPOINT:
    LLM_TRUSS_MODELS = {
        TRUSS_LLM_LLAMA2_CHAT: LLM_TRUSS_MODEL_ENDPOINT,
    }
    LLM_TYPES.extend(LLM_TRUSS_MODELS)

Logger.info(f"LLM types loaded {LLM_TYPES}")

DEFAULT_QUERY_CHAT_MODEL = VERTEX_LLM_TYPE_BISON_CHAT

# embedding models
VERTEX_EMBEDDING_MODELS = [VERTEX_LLM_TYPE_GECKO_EMBEDDING]
LANGCHAIN_EMBEDDING_MODELS = []
if ENABLE_OPENAI_LLM:
  LANGCHAIN_EMBEDDING_MODELS.append(OPENAI_EMBEDDING_TYPE)
  LANGCHAIN_LLM.update({
    OPENAI_EMBEDDING_TYPE: OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
  })

EMBEDDING_MODELS = VERTEX_EMBEDDING_MODELS + LANGCHAIN_EMBEDDING_MODELS
DEFAULT_QUERY_EMBEDDING_MODEL = VERTEX_LLM_TYPE_GECKO_EMBEDDING

# services config
SERVICES = {
  "user-management": {
    "host": "http://user-management",
    "port": 80,
    "api_path": "/user-management/api/v1",
    "api_url_prefix": "http://user-management:80/user-management/api/v1",
  },
  "tools-service": {
    "host": "http://tools-service",
    "port": 80,
    "api_path": "/tools-service/api/v1",
    "api_url_prefix": "http://tools-service:80/tools-service/api/v1",
  },
  "rules-engine": {
    "host": "http://rules-engine",
    "port": 80,
    "api_path": "/rules-engine/api/v1",
    "api_url_prefix": "http://rules-engine:80/rules-engine/api/v1",
  }
}

USER_MANAGEMENT_BASE_URL = f"{SERVICES['user-management']['host']}:" \
                  f"{SERVICES['user-management']['port']}" \
                  f"/user-management/api/v1"

TOOLS_SERVICE_BASE_URL = f"{SERVICES['tools-service']['host']}:" \
                  f"{SERVICES['tools-service']['port']}" \
                  f"/rules-engine/api/v1"

RULES_ENGINE_BASE_URL = f"{SERVICES['rules-engine']['host']}:" \
                  f"{SERVICES['rules-engine']['port']}" \
                  f"/rules-engine/api/v1"

try:
  LLM_BACKEND_ROBOT_USERNAME = get_secret("llm-backend-robot-username")
except Exception as e:
  LLM_BACKEND_ROBOT_USERNAME = None

try:
  LLM_BACKEND_ROBOT_PASSWORD = get_secret("llm-backend-robot-password")
except Exception as e:
  LLM_BACKEND_ROBOT_PASSWORD = None

# Update this config for local development or notebook usage, by adding
# a parameter to the UserCredentials class initializer, to
# pass URL to auth client.
# auth_client = UserCredentials(LLM_BACKEND_ROBOT_USERNAME,
#                               LLM_BACKEND_ROBOT_PASSWORD,
#                               "http://localhost:9004")
# pass URL to auth client for external routes to auth.  Replace dev.domain with
# the externally mapped domain for your dev server
# auth_client = UserCredentials(LLM_BACKEND_ROBOT_USERNAME,
#                               LLM_BACKEND_ROBOT_PASSWORD,
#                               "https://[dev.domain]")

auth_client = UserCredentials(LLM_BACKEND_ROBOT_USERNAME,
                              LLM_BACKEND_ROBOT_PASSWORD)

# agent config
AGENT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "agent_config.json")
