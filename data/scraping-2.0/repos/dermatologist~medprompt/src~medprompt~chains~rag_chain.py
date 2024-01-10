"""
 Copyright 2023 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import logging
from typing import List

from kink import di
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.tools import tool
from langchain.vectorstores import FAISS, Chroma, Redis
from langchain_core.pydantic_v1 import BaseModel, Field
from overrides import override

from ..tools.create_embedding import CreateEmbeddingFromFhirBundle

_logger = logging.getLogger(__name__)

from .base_medprompt_chain import BaseMedpromptChain


class RagChain(BaseMedpromptChain):

    class RagChainInput(BaseModel):
        current_patient_context: str = Field(default="")
        input: str = Field()
        patient_id: str = Field()

    def __init__(self,
                 chain=None,
                 prompt={},
                 main_llm=None,
                 clinical_llm=None,
                 sec_llm=None,
                 input_type=None,
                 output_type=None
                 ):
        super().__init__(
                chain = chain,
                prompt = prompt,
                main_llm = main_llm,
                clinical_llm = clinical_llm,
                sec_llm = sec_llm,
                input_type = input_type,
                output_type = output_type,
                )
        self._input_type = self.RagChainInput
        self._name = "extended search"
        self.init_prompt()

    @override
    def init_prompt(self):
        QUESTION_TEMPLATE = """Given the following chat history and a follow up question, rephrase the
        follow up question to be a standalone question, in its original language that includes context from chat history below.

        Chat History:
        {current_patient_context}
        Follow Up Question: {input}
        Standalone question:"""

        if "QUESTION_TEMPLATE" in self._prompt:
            QUESTION_TEMPLATE = self._prompt["QUESTION_TEMPLATE"]

        self.CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(QUESTION_TEMPLATE)

        ANSWER_TEMPLATE = """Answer the following question based only on the available context below.

        Context:
        {context}

        Question:
        {input}

        """

        if "ANSWER_TEMPLATE" in self._prompt:
            ANSWER_TEMPLATE = self._prompt["ANSWER_TEMPLATE"]

        self.ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

        DEFAULT_TEMPLATE = """{page_content}"""

        if "DEFAULT_TEMPLATE" in self._prompt:
            DEFAULT_TEMPLATE = self._prompt["DEFAULT_TEMPLATE"]

        self.DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template=DEFAULT_TEMPLATE)

        self.EMBED_MODEL = di["embedding_model"]
        self.INDEX_SCHEMA = di["index_schema"]
        self.REDIS_URL = di["redis_url"]
        self.embedding = HuggingFaceEmbeddings(model_name=self.EMBED_MODEL)
        self.VECTORSTORE_NAME = di["vectorstore_name"]

    def check_index(self, input_object):
        """Check if the index exists for the patient."""
        patient_id = input_object["patient_id"]
        if self.VECTORSTORE_NAME == "redis":
            create_embedding_tool = CreateEmbeddingFromFhirBundle()
            _ = create_embedding_tool.run(patient_id)
            vectorstore = Redis.from_existing_index(
                embedding=self.embedding, index_name=patient_id, schema=self.INDEX_SCHEMA, redis_url=self.REDIS_URL
            )
        elif self.VECTORSTORE_NAME == "chroma":
            create_embedding_tool = CreateEmbeddingFromFhirBundle()
            _ = create_embedding_tool.run(patient_id)
            vectorstore = Chroma(collection_name=patient_id, persist_directory=di["vectorstore_path"], embedding_function=self.embedding)
            vectorstore.persist()
        elif self.VECTORSTORE_NAME == "faiss":
            create_embedding_tool = CreateEmbeddingFromFhirBundle()
            _ = create_embedding_tool.run(patient_id)
            fname = di["vectorstore_path"] + "/" + patient_id + ".index"
            vectorstore = FAISS.load_local(fname, embeddings=self.embedding)
        return vectorstore.as_retriever().get_relevant_documents(input_object["input"], k=10)


    def _combine_documents(
        self,
        docs, document_separator="\n\n"
    ):
        """Combine documents into a single string."""
        try:
            doc_strings = [format_document(doc, self.DEFAULT_DOCUMENT_PROMPT) for doc in docs]
            _reply = document_separator.join(doc_strings)
            if len(_reply.strip()) < 3:
                _reply = "No information found. The vectorstore may still be indexing. Please try again later."
        except:
            _reply = "No information found. The vectorstore may still be indexing. Please try again later."
        return _reply


    # def _format_current_patient_context(self, current_patient_context: List[str]) -> str:
    #     """Format chat history into a string."""
    #     buffer = ""
    #     if not current_patient_context:
    #         return buffer
    #     for dialogue_turn in current_patient_context:
    #         buffer += "\n" + dialogue_turn
    #     return buffer


    @property
    @override
    def chain(self):
        """Get the runnable chain."""
        context = RunnablePassthrough.assign(
            current_patient_context=lambda x: x.get("current_patient_context", ""),
            patient_id=lambda x: x["patient_id"],
            input=lambda x: x["input"],
        ) | self.check_index | self._combine_documents
        input = RunnablePassthrough.assign(
            current_patient_context=lambda x: x.get("current_patient_context", ""),
            patient_id=lambda x: x["patient_id"],
            input=lambda x: x["input"],
        ) | self.CONDENSE_QUESTION_PROMPT | self.main_llm | StrOutputParser()
        _inputs = RunnableMap(
            context=context,
            input=input,
        )
        _chain = _inputs | self.ANSWER_PROMPT | self.clinical_llm | StrOutputParser()
        chain = _chain.with_types(input_type=self.input_type)
        return chain


@tool(RagChain().name, args_schema=RagChain().input_type)
def get_tool(**kwargs):
    """
    Searches lab tests, investigations, clinical notes, reports and diagnostic panels.

    The input is a dict with the following mandatory keys:
        patient_id (str): The id of the patient to search for.
        input (str): The question to ask the model based on the available context.
        current_patient_context (str): The previous conversation history.
    """
    return RagChain().chain.invoke(kwargs)