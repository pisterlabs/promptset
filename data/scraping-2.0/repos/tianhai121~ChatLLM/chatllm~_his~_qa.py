#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : AI.  @by PyCharm
# @File         : qa
# @Time         : 2023/4/20 17:47
# @Author       : betterme
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :

from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS

# ME
from chatllm._his._chatllm import ChatLLM

RetrievalQA.return_source_documents = True


class QA(object):
    def __init__(self, chatllm: ChatLLM, faiss_ann: FAISS = None, document_prompt: PromptTemplate = None):
        """

        :param chatllm:
        """
        self.chatllm = chatllm
        self.faiss_ann = faiss_ann
        self.document_prompt = document_prompt if document_prompt else self.default_document_prompt

    @property
    def default_document_prompt(self) -> PromptTemplate:
        prompt_template = """
        基于以下已知信息，简洁和专业的来回答用户的问题。
        如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
        已知内容:
        {context}
        问题:
        {question}
        """.strip()

        return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    def get_knowledge_based_answer(self, query, max_turns=3, top_k=4, **kwargs):
        assert self.faiss_ann

        # 设置chatllm参数，# history会被储存？
        self.chatllm.set_chat_kwargs(**kwargs)
        self.chatllm.max_turns = max_turns

        llm_chain = RetrievalQA.from_llm(
            llm=self.chatllm,
            retriever=self.faiss_ann.as_retriever(search_kwargs={"k": top_k}),  # todo: 重复实例化优化
            prompt=self.document_prompt
        )
        # llm_chain.combine_documents_chain.document_prompt = PromptTemplate(
        #     input_variables=["page_content"], template="{page_content}"
        # )
        # 官方默认，要不要覆盖
        # document_prompt = PromptTemplate(
        #     input_variables=["page_content"], template="Context:\n{page_content}"
        # )

        result = llm_chain({"query": query})
        return result

    def get_llm_answer(self, query, max_turns=3, **kwargs):  # 重复代码
        self.chatllm.set_chat_kwargs(**kwargs)
        self.chatllm.max_turns = max_turns

        return self.chatllm._call(query)

