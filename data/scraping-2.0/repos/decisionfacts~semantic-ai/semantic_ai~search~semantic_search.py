import asyncio
from typing import Optional

import torch
import logging

from fastapi import HTTPException, status
from semantic_ai.utils import sync_to_async, _clear_cache
from semantic_ai.constants import DEFAULT_PROMPT

from langchain.chains import RetrievalQA
from langchain import PromptTemplate

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Search:

    def __init__(self,
                 model,
                 load_vector_db,
                 top_k: Optional[int] = None,
                 prompt: Optional[str] = None
                 ):
        self.model = model
        self.load_vector_db = load_vector_db
        self.top_k = top_k or 4
        self.prompt_template = prompt or DEFAULT_PROMPT

    async def generate(self, query: str):
        asyncio.create_task(_clear_cache())
        with (torch.inference_mode()):
            _no_response = "Sorry, I can't find the answer from the document."
        prompt_template = PromptTemplate(template=self.prompt_template,
                                         input_variables=["context", "question"])
        chain_type_kwargs = {
            "prompt": prompt_template
        }
        vector_search = self.load_vector_db
        # print(f"Search Query: {vector_search.similarity_search(query)}")
        retriever = await sync_to_async(
            vector_search.as_retriever,
            search_kwargs={"k": self.top_k}
        )
        qa: RetrievalQA = await sync_to_async(
            RetrievalQA.from_chain_type,
            llm=self.model,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        try:
            result = await sync_to_async(qa, query)
            if result:
                # print("Retrieval Result =:\n", result)
                logger.info(f"Retrieval Result =:\n{result}")
                source_documents = [doc.metadata for doc in result.get('source_documents') or []]
                llm_result = result.get('result')
                llm_response = {'query': query, 'result': llm_result,
                                'source_documents': source_documents}
                asyncio.create_task(_clear_cache())
                return llm_response
            else:
                null_response = {'query': query, 'result': _no_response}
                asyncio.create_task(_clear_cache())
                return null_response
        except Exception as ex:
            logger.error('Vector Query call error!=> ', exc_info=ex)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Sorry! No response found."
            )
