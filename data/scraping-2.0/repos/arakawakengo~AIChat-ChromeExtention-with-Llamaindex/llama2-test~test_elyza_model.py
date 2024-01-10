import logging
import sys

# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct"
)
model = AutoModelForCausalLM.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)


from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# パイプラインの準備
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

# LLMの準備
llm = HuggingFacePipeline(pipeline=pipe)


from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from typing import Any, List

# 埋め込みクラスにqueryを付加
class HuggingFaceQueryEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(["query: " + text for text in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query("query: " + text)

# 埋め込みモデルの準備
embed_model = LangchainEmbedding(
    HuggingFaceQueryEmbeddings(model_name="intfloat/multilingual-e5-large")
)


from llama_index import ServiceContext
from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser

# ノードパーサーの準備
text_splitter = SentenceSplitter(
    chunk_size=300,
    paragraph_separator="\n\n",
    tokenizer=tokenizer.encode,
    chunk_overlap=100
)
node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)

# サービスコンテキストの準備
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    node_parser=node_parser,
)


from llama_index import SimpleDirectoryReader

# ドキュメントの読み込み
documents = SimpleDirectoryReader(
    input_files=["/home/Nikkei-intern/intern2023-kyoto-team-basilico/llama2-test/data/sample.txt"]
).load_data()



from llama_index import VectorStoreIndex

# インデックスの作成
index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
)


from llama_index.prompts.prompts import QuestionAnswerPrompt

# QAテンプレートの準備
qa_template = QuestionAnswerPrompt("""<s>[INST] <<SYS>>
質問に100文字以内で答えだけを回答してください。
<</SYS>>
{query_str}

{context_str} [/INST]
""")


# クエリエンジンの作成
query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_template,
)


# 質問応答
response = query_engine.query("習近平ってどんな人？")

print(response)
