import os
import sys
from typing import List

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

from utils.model_config import embedding_model_dict
from utils.predictor import ChatGLMPredictor
from utils.utils import torch_gc, load_file, similarity_search_with_score_by_vector, get_docs_with_score, \
    generate_prompt


class LocalDocQA:

    def __init__(self, llm_model: ChatGLMPredictor, embedding_model: str = 'ernie-base', vs_path: str = "vector_store/",
                 top_k=5, chunk_size=250, device="cuda", cache_dir='cache/'):
        """
        本地知识问答器

        :param llm_model: 语言模型
        :param embedding_model: 词向量编码模型名称
        :param vs_path: 知识文件词向量文件夹
        :param top_k: 获取指定数量的句子和相似度的词汇表
        :param chunk_size: 词向量的分片大小
        :param device: 指定使用推理设备
        :param cache_dir: 词向量模型的缓存路径
        """
        self.llm_model = llm_model
        self.vs_path = vs_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.vector_store = None
        # 判断是否存在的模型
        assert embedding_model in embedding_model_dict.keys(), \
            f"{embedding_model} not found in {list(embedding_model_dict.keys())}"
        # 获取词向量编码模型
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                cache_folder=cache_dir, model_kwargs={'device': device})
        # 加载词典
        if os.path.exists(self.vs_path):
            self.vector_store = FAISS.load_local(self.vs_path, self.embeddings)
            FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
            self.vector_store.chunk_size = self.chunk_size

    def init_knowledge_vector_store(self, filepath: str or List[str]):
        """加载知识表，并将其存储到词典中，之后可以直接使用，不需要再次加载

        :param filepath: 本地知识文件路径
        :return:
        """
        assert os.path.exists(filepath), f"{filepath}路径不存在"
        docs = []
        if isinstance(filepath, str):
            if os.path.isfile(filepath):
                try:
                    docs = load_file(filepath)
                    print(f"{filepath} 已成功加载")
                except Exception as e:
                    print(f"{filepath} 未能成功加载，错误信息：{e}")
            elif os.path.isdir(filepath):
                for file in tqdm(os.listdir(filepath), desc="加载文件"):
                    file_path = os.path.join(filepath, file)
                    try:
                        docs += load_file(file_path)
                    except Exception as e:
                        print(f"{file} 未能成功加载，错误信息：{e}")
        elif isinstance(filepath, list):
            for file in tqdm(filepath, desc="加载文件"):
                try:
                    docs += load_file(file)
                except Exception as e:
                    print(f"{file} 未能成功加载，错误信息：{e}")

        if len(docs) > 0:
            print("文件加载完毕，正在生成向量库，这个过程时间可能有点久，请耐心等待...")
            if self.vector_store is not None:
                self.vector_store.add_documents(docs)
            else:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            torch_gc()
            self.vector_store.save_local(self.vs_path)
        else:
            print("全部文件均未成功加载，请检查依赖包或替换为其他文件再次上传！", file=sys.stderr)

    def get_knowledge_based_answer(self,
                                   query,
                                   streaming=True,
                                   session_id=None,
                                   **kwargs):
        """使用本地知识文件和语言模型进行回答问题

        :param query: 用户输入的问题
        :param streaming: 是否支持流式回答
        :param session_id: 对话的会话ID
        :param kwargs: 语言模型的其他参数
        :return: 模型输出的结果和会话ID
        """
        assert self.vector_store, f"文本向量为None"
        related_docs_with_score = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        torch_gc()
        prompt = generate_prompt(related_docs, query)
        print(prompt)
        if streaming:
            for output in self.llm_model.generate_stream(prompt=prompt, session_id=session_id, **kwargs):
                torch_gc()
                session_id = output['session_id']
                result = output['response']

                response = {"query": query,
                            "result": result,
                            "source_documents": related_docs}
                yield response, session_id
                torch_gc()
        else:
            output = self.llm_model.generate(prompt=prompt, session_id=session_id, **kwargs)
            torch_gc()
            session_id = output['session_id']
            result = output['response']

            response = {"query": query,
                        "result": result,
                        "source_documents": related_docs}
            torch_gc()
            return response, session_id
