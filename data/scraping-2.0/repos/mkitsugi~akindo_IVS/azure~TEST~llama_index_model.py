import os

import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from llama_index import (
    Document,
    LangchainEmbedding,
    LLMPredictor,
    PromptHelper,
    QuestionAnswerPrompt,
    ServiceContext,
    SimpleWebPageReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.prompts.chat_prompts import (
    CHAT_REFINE_PROMPT,
    CHAT_REFINE_PROMPT_TMPL_MSGS,
)
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
from llama_index.prompts.prompts import RefinePrompt

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# インデックスの読み込み
storage_context = StorageContext.from_defaults(persist_dir="azure/TEST/strage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

# クエリを投げる
qestion = "好きな食べ物は？"
response = query_engine.query(qestion)
print(response)
