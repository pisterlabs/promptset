from threading import RLock
# 重写继承模型
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# 提示词模板
from langchain.prompts import PromptTemplate
# 编码模型及向量数据库
from embeddings.embedding_helper import EmbeddingHelper
from typing import List
from llms.llm_base_model import ChatGLM
from configs.base_config import BASE_CONFIG
from llms.llm_gpt_model import ChatGPT


class ChatGLM_Helper(object):
    single_lock = RLock()
    model_id: str = BASE_CONFIG["llm_model_path"]
    is_local_model: bool = True

    def __init__(self, **kwargs) -> None:
        if "model_id" in kwargs:
            self.model_id = kwargs["model_id"]

        self.llm = ChatGPT()
        self.llm.load_model()

    # 给予用户输入构建提示词（用于知识中心问答）
    def build_prompt(
            self,
            output_parsers: StructuredOutputParser = None) -> PromptTemplate:
        template = """已知信息:'{context}'。问题:'{question}'。
        请根据已知信息，简洁和专业地使用中文回答用户的问题。
        如果无法从已知信息中得到答案，请说 "根据已知信息无法回答该问题"，
        不允许在答案中添加编造成分。"""
        if output_parsers:
            template += "\n {format_instructions}"
            format_instructions = output_parsers.get_format_instructions()
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
                partial_variables={"format_instructions": format_instructions})
            return prompt
        else:
            prompt = PromptTemplate(input_variables=["context", "question"],
                                    template=template)
            return prompt

    # 构建输出格式
    def build_output_parsers(
            self, schemas: List[ResponseSchema]) -> StructuredOutputParser:
        if not schemas:
            return None
        response_schemas = []
        for item in schemas:
            response_schemas.append(item)
        output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas)
        return output_parser

    # 给予用户输入查询该输入的上线文相关性描述
    def query_context(self, input: str, k: int = 2):
        helper = EmbeddingHelper()
        result = helper.query(input, k)
        db_context = ""
        if len(result) > 0:
            db_context = "\n".join(result)
        return db_context

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(ChatGLM_Helper, "_instance"):
            with ChatGLM_Helper.single_lock:
                if not hasattr(ChatGLM_Helper, "_instance"):
                    ChatGLM_Helper._instance = cls(*args, **kwargs)
        return ChatGLM_Helper._instance
