import re
from typing import Dict, List, Union, Optional

import numpy as np
import langchain
from langchain.docstore.document import Document
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI, BaseLLM

from thinkgpt.helper import get_n_tokens, fit_context

EXECUTE_WITH_CONTEXT_PROMPT = PromptTemplate(template="""
Given a context information, reply to the provided request
Context: {context}
User request: {prompt}
""", input_variables=["prompt", "context"], )



class ExecuteWithContextChain(LLMChain):
    """Prompts the LLM to execute a request with potential context"""
    def __init__(self, **kwargs):
        super().__init__(prompt=EXECUTE_WITH_CONTEXT_PROMPT, **kwargs)


class MemoryMixin:
    memory: langchain.vectorstores
    mem_cnt: int
    embeddings_model: OpenAIEmbeddings

    def memorize(self,
                 concept: Union[str, Document, List[Document]]):
        ''' Memorize some data by saving it to a vectorestore like Chroma
        '''
        self.mem_cnt += 1
        #---------------------------------------
        # Create the documents to store
        #---------------------------------------
        if isinstance(concept, str):
            docs = [Document(page_content=concept)]
        elif isinstance(concept, Document):
            docs = [concept]
        elif isinstance(concept, list):
            docs = concept[:]
            if any([not isinstance(con, Document) for con in concept]):
                raise ValueError('wrong type for List[Document]')
        else:
            raise ValueError('wrong type, must be either str, Document, List[Document]')

        #---------------------------------------
        # Save memory into the database
        #---------------------------------------
        self.memory.add_documents(docs)
        return None

    def remember(self,
                 concept: str,
                 limit: int = 5,
                 max_tokens: Optional[int] = None) -> List[str]:
        #------------------------------------------------------
        # Cannot remember if there is no stored memories
        #------------------------------------------------------
        if len(self.memory) == 0:
            return []
        #------------------------------------------------------
        # Grab the most relevant memories
        # memory needs to be sorted in chronological order
        #------------------------------------------------------
        docs = self.memory.similarity_search(concept, k=limit)
        text_results = [doc.page_content for doc in docs]
        if max_tokens:
            text_results = fit_context(text_results, max_tokens)
        return text_results
