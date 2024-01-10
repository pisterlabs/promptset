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
from llama_index.prompts.chat_prompts import (
    CHAT_REFINE_PROMPT,
    CHAT_REFINE_PROMPT_TMPL_MSGS,
)
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
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


# データの準備 / 本番はDBから取ってくる想定
data = [
    "そうなんだよね、最近困ってて…",
    "仕事が忙しくて、彼女も見つけられないんだよねー。いい人いないかな？優しくて明るい人！",
    "そうそう、趣味とかはゴルフやってるんだ〜♪",
    "僕の趣味覚えてるー？",
    "僕の名前はたくやです！",
    "あれ、相談乗ってくれないの？🥺",
    "僕の名前を覚えてくれる？拓也って言うんだけど",
    "僕の名前を覚えてるか教えて？",
    "僕の趣味は？",
    "僕の趣味覚えてる？",
    "僕のの好きなものは唐揚げなんだー",
]

# List[Document]に変形する
documents = [Document(text=item) for item in data]

# インデックスを作成する
index = VectorStoreIndex.from_documents(documents, service_context=service_context) #ベクトルデータの生成
index.storage_context.persist("azure/TEST/strage") #AzureのDBに投げる

# クエリエンジンを生成する
query_engine = index.as_query_engine(
    service_context=service_context, text_qa_template=QA_PROMPT, refine_template=CHAT_PROMPT
)

# クエリを投げる
qestion = "好きな食べ物は？"
response = query_engine.query(qestion)
print(response)
# >>> 唐揚げです。
print(response.get_formatted_sources(length=4096))

#質問に対する回答
#データをベクトルデータに変更