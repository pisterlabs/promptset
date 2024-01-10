"""Chain for chatting with a vector database."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from langchain.chains import ChatVectorDBChain
from typing import Optional

from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.combine_documents.stuff import StuffDocumentsChain 
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import stuff_prompt
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

from pydantic import BaseModel
from langchain.docstore.document import Document

def unique_doc_content(docs):
    """Check whether the content of the docs are the same or not and remain the unique ones"""
    unique_docs = []
    seen_page_content = set()
    for doc in docs:
        if doc.page_content not in seen_page_content:
            seen_page_content.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs

class ChatEBAVectorDBChain(ChatVectorDBChain):
    """Chain for chatting with a vector database customized for EBA"""
    threshold = 0.35

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs["question"]
        new_question = question
        docs = self.vectorstore.similarity_search_with_score(new_question, k=4)
        docs = [doc[0] for doc in docs if doc[1] > self.threshold]
        #Eliminate duplicate doc if the doc.content is the same
        docs = unique_doc_content(docs)

        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        answer, _ = self.combine_docs_chain.generate(docs, **new_inputs)
        return {self.output_key: answer}

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs["question"]
        new_question = question
        docs = self.vectorstore.similarity_search_with_score(new_question, k=4)
        ## Flush the docs if their score is too low
        docs = [doc[0] for doc in docs if doc[1] > self.threshold]
        ## Check whether the content of the docs are the same or not and remain the unique ones
        docs = unique_doc_content(docs)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        answer, _ = await self.combine_docs_chain.agenerate(docs,  **new_inputs)
        return {self.output_key: answer}

class FewShotStuffChain(StuffDocumentsChain, BaseModel):

    fewshots_prompt: FewShotPromptTemplate
    few_shot_variable_name: str
    
    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        doc_dicts = []
        for doc in docs:
            base_info = {"page_content": doc.page_content}
            base_info.update(doc.metadata)
            document_info = {
                k: base_info[k] for k in self.document_prompt.input_variables
            }
            doc_dicts.append(document_info)

        doc_strings = [self.document_prompt.format(**doc) for doc in doc_dicts]
        samples_string = self.fewshots_prompt.format(question=kwargs["question"])
        inputs = kwargs.copy()
        inputs[self.document_variable_name] = "\n\n".join(doc_strings)
        ##Ugly hacking code for few-shot prompt with EBA prompt
        inputs[self.few_shot_variable_name] = "\n" + samples_string.replace(kwargs["question"], "")

        return inputs
    
    def prompt_length(self, docs: List[Document], **kwargs: Any) -> Optional[int]:
        """Get the prompt length by formatting the prompt."""
        inputs = self._get_inputs(docs, **kwargs)
        prompt = self.llm_chain.prompt.format(**inputs)
        return self.llm_chain.llm.get_num_tokens(prompt)

    def generate(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        """Stuff all documents and few-shot examples into one prompt and pass to LLM."""
        inputs = self._get_inputs(docs, **kwargs)
        return self.llm_chain.predict(**inputs), {}

    async def agenerate(
        self, docs: List[Document], **kwargs: Any
    ) -> Tuple[str, dict]:
        """Stuff all documents and few-shot examples into one prompt and pass to LLM."""
        inputs = self._get_inputs(docs, **kwargs)
        prompt = self.llm_chain.prompt.format(**inputs)
        print(prompt)
        return await self.llm_chain.apredict(**inputs), {}

def load_qaf_chain(
    llm: BaseLLM,
    fewshots_prompt: FewShotPromptTemplate,
    prompt: BasePromptTemplate = stuff_prompt.PROMPT,
    document_variable_name: str = "context",
    few_shot_variable_name: str = "few_shot_samples",
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> FewShotStuffChain:

    llm_chain = LLMChain(
        llm=llm, prompt=prompt, verbose=verbose, callback_manager=callback_manager
    )
    few_stuff_chain = FewShotStuffChain(
        llm_chain=llm_chain,
        fewshots_prompt=fewshots_prompt,
        document_variable_name=document_variable_name,
        few_shot_variable_name=few_shot_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )
    return few_stuff_chain