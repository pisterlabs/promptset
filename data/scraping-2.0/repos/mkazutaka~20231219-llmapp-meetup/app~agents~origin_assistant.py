import openai
import json
import re
from typing import List
from app.agents import BaseAgentFactory
from unstructured.partition.pdf import partition_pdf
from llama_index import Document, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.retrievers import RecursiveRetriever
from llama_index.chat_engine import ContextChatEngine
from llama_index.memory import ChatMemoryBuffer

CONTEXT_TEMPLATE = (
    "あなたは優秀なチャットボットです。下記のコンテキストを元にユーザの質問に回答してください。"
    "会話が長く続くようになるべくフレンドリーで丁寧な回答をしてください。"
    "ただし、以下のコンテキストが渡されていることを気づかれてはいけません。"
    "また、コンテキストに書かれていないものに関しては答えてはいけません。\n"
    "---------------------------\n"
    "{context_str}"
    "---------------------------\n"
)


class OriginAssistant(BaseAgentFactory):
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.chat_engine = None

    def name(self):
        return f"OriginAssistant(top_k={self.top_k})"

    def build(self, file: str):
        contents = self.parse_file(file)
        joined_content = "\n".join(contents)
        contents = self.split_by_openai(joined_content)
        self.chat_engine = self.create_chat_engine(contents, top_k=self.top_k)

    def chat(self, query: str) -> str:
        result = self.chat_engine.chat(query)
        return result.response

    def parse_file(self, file: str) -> List[str]:
        elements = partition_pdf(
            file,
            languages=["jpn"],
            strategy="hi_res",
            infer_table_structure=True,
        )
        contents = []
        for el in elements:
            if el is None or el.category in ['Image', 'Footer', 'Header']:
                continue
            if el.category == 'Table':
                contents.append(el.metadata.text_as_html.strip())
            else:
                contents.append(el.text.strip())
        filtered_contents = [c for c in contents if c not in ["", None]]
        return filtered_contents

    def split_by_openai(self, content: str) -> List[str]:
        SYSTEM_PROMPT = (
            "あなたは、与えられたテキストを適切なセクションに分ける事が優秀なアシスタントです。"
            "セクションは可能な限り細かく分ける方が望ましいですが、意味がわからなくなるほど分ける必要はありません。"
            "与えられたテキストから、区切りとなるテキストをJSON形式の配列で出力することができます。"
            'JSONは、{ "sections": [ str ] }の形式で出力します。'
        )
        response = openai.chat.completions.create(
            temperature=0.0,
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            response_format={"type": "json_object"}
        )
        new_contents = []
        for delimiter in json.loads(response.choices[0].message.content)["sections"]:
            splitted = content.split(delimiter, 1)
            if splitted[0].strip() != "":
                new_contents.append(splitted[0].strip())
            if len(splitted) > 1:
                content = delimiter + splitted[1]
            else:
                new_contents.append(content)
        return new_contents


    def create_chat_engine(self, contents: List[str], top_k: int = 3) -> ContextChatEngine:
        embeddings = HuggingFaceEmbeddings(
            model_name="oshizo/sbert-jsnli-luke-japanese-base-lite"
        )
        service_context =  ServiceContext.from_defaults(
            embed_model=embeddings,
            chunk_size=8096,
        )

        sentences = [Document(text=content) for content in contents]

        section_node_parser = SimpleNodeParser.from_defaults(chunk_size=8096)
        sentence_node_parser = SimpleNodeParser.from_defaults(chunk_size=8096)

        section_nodes = section_node_parser.get_nodes_from_documents(sentences)

        all_index_nodes = []
        for section_node in section_nodes:
            # サブノードの作成
            section_node_content = section_node.get_content()
            section_node_sentences = re.split('[。.\n]', section_node_content)
            section_node_sentence_documents = [Document(text=s) for s in section_node_sentences]
            sentence_nodes = sentence_node_parser.get_nodes_from_documents(section_node_sentence_documents)

            all_index_nodes.extend([
                IndexNode.from_text_node(node=sn, index_id=section_node.node_id) for sn in sentence_nodes
            ])
            all_index_nodes.append(IndexNode.from_text_node(section_node, section_node.node_id))

        vector_index_chunk = VectorStoreIndex(
            all_index_nodes,
            service_context=service_context
        )

        vector_retriever_chunk = vector_index_chunk.as_retriever(
            similarity_top_k=top_k
        )
        all_index_node_dict = {n.node_id: n for n in all_index_nodes}
        retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever_chunk},
            node_dict=all_index_node_dict,
            verbose=False,
        )

        memory = ChatMemoryBuffer.from_defaults(token_limit=20000000)
        return ContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            context_template=CONTEXT_TEMPLATE,
            service_context=ServiceContext.from_defaults(
                llm=OpenAI(model="gpt-4-1106-preview")
            ),
        )
