import json
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
    QuestionAnswerPrompt,
    ServiceContext,
    VectorStoreIndex,
)
from llama_index.prompts.chat_prompts import CHAT_REFINE_PROMPT_TMPL_MSGS
from llama_index.prompts.prompts import RefinePrompt

# デフォルトのプロンプトを変更
QA_PROMPT_TMPL = (
    "以下の情報を参照してください。 \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "この情報を使って、次の質問に回答してください。: {query_str}\n"
)

CHAT_REFINE_PROMPT_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template("{query_str}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(
        """
    以下の情報を参照してください。 \n"
    "---------------------\n"
    "{context_msg}"
    "\n---------------------\n"
    この情報が回答の改善に役立つようならこの情報を使って回答を改善してください。
    この情報が回答の改善に役立たなければ元の回答を日本語で返してください。
    """
    ),
]

CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)

QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
CHAT_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# デフォルトと同じだが一応明示する
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
llm_predictor = LLMPredictor(llm)
llama_embed = LangchainEmbedding(
    embedding,
    embed_batch_size=1,
)
# エンベディングモデルをロードする
# この例では、デフォルトのエンベディングモデルを使用しています
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=llama_embed,
)

# Load data
with open("azure/TEST/sample_1.txt", "r") as f:
    # data = f.readlines()
    content = f.read().strip()
    data = json.loads(content)
messages = []
# messagesにメッセージを格納
for item in data:
    message = []
    for user in item:
        if user["user_id"] != "AI":
            message.append(user["message"])
    messages.append(message)

# List[Document]に変形する
documents_1 = [Document(text=item) for item in messages[0]]
documents_2 = [Document(text=item) for item in messages[1]]
documents_3 = [Document(text=item) for item in messages[2]]

# インデックスを作成する
index_1 = VectorStoreIndex.from_documents(documents_1, service_context=service_context)
index_1.storage_context.persist("azure/TEST/strage_1")

# インデックスを作成する
index_2 = VectorStoreIndex.from_documents(documents_2, service_context=service_context)
index_2.storage_context.persist("azure/TEST/strage_2")

# インデックスを作成する
index_3 = VectorStoreIndex.from_documents(documents_3, service_context=service_context)
index_3.storage_context.persist("azure/TEST/strage_3")

# クエリエンジンを生成する
query_engine = index_1.as_query_engine(
    service_context=service_context, text_qa_template=QA_PROMPT, refine_template=CHAT_PROMPT
)

# クエリを投げる
qestion = "好きな食べ物は？"
response = query_engine.query(qestion)
print(response)
# >>> 唐揚げです。
print(response.get_formatted_sources(length=4096))

# userのチャットメッセージだけ取ってくる
# llama_indexにコピペで投げている
