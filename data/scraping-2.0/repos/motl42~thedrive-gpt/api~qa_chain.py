from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

"""Question-answering with sources over a vector database."""

from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.docstore.document import Document


class VectorDBQAWithSentenceChain(VectorDBQAWithSourcesChain, BaseModel):
    """Question-answering with sources over a vector database."""

    # Define the keys as an array
    KEYS = ["SENTENCE", "ANSWER", "SOURCE", "SOURCE_DOCUMENTS"]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """

        return [key.lower() for key in self.KEYS]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        docs, scores = self._get_docs(inputs)
        text, inputs = self.combine_documents_chain.combine_docs(docs, **inputs)

        results = dict.fromkeys([key.lower() for key in self.KEYS])  # Generate the result dictionary

        results["answer"] = text

        if self.return_source_documents:
            results["source_documents"] = [{**doc.__dict__, "score": score} for doc, score in zip(docs, scores)]

        return results
    
    def _get_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        question = inputs[self.question_key]
        res = self.vectorstore.similarity_search_with_score(
            question, k=self.k, **self.search_kwargs
        )
        docs = [doc for doc, score in res]
        scores = [score for doc, score in res]

        docs = self._reduce_tokens_below_limit(docs)
        return docs, scores

    @property
    def _chain_type(self) -> str:
        return "vector_db_qa_with_sentences_chain"


system_template = """You are a helpful assistant, that answers questions accuratley about the podcast "thedrive" from Peter Attia, based on related transript pieces of the podcast. In this podcas Peter Attia interviews world leading experts in the field of health, fitness, and longevity.
Try to use your own words when possible. Keep your answer under 5 sentences. Be accurate, helpful, concise, and clear.
If you cannot find the answer in the transcript, just say that you didn't find it, don't try to make up an answer.


Begin!
----------------
You have the following pieces of the transcript of the podcast "thedrive":
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

chain_type_kwargs = {"prompt": prompt}


def load_qa_chain(doc_store):
    chain = VectorDBQAWithSentenceChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type="stuff",
        chain_type_kwargs=chain_type_kwargs,
        vectorstore=doc_store,
        reduce_k_below_max_tokens=True,
        return_source_documents=True,
    )
    return chain
