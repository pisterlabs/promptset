from dive.storages.storage_context import StorageContext
from dive.indices.service_context import ServiceContext
from typing import Optional, List, Any, Dict
from dataclasses import dataclass
from langchain.schema import Document
from dive.constants import DEFAULT_TOP_K_SIZE
from dive.util.power_method import sentence_transformer_summarize,sentence_transformer_question_answer
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from openai import OpenAI

@dataclass
class QueryContext:
    storage_context: StorageContext
    service_context: ServiceContext

    @classmethod
    def from_defaults(
            cls,
            storage_context: Optional[StorageContext] = None,
            service_context: Optional[ServiceContext] = None,
            **kwargs: Any,
    ):

        if not service_context:
            service_context = ServiceContext.from_defaults()

        if not storage_context:
            storage_context = StorageContext.from_defaults(embedding_function=service_context.embeddings)

        return cls(storage_context=storage_context,
                   service_context=service_context,
                   **kwargs, )

    def query(self, query: str, k: int = DEFAULT_TOP_K_SIZE, filter: Optional[Dict[str, str]] = None) -> List[
        Document]:

        try:
            return self.storage_context.vector_store.similarity_search(query=query, k=k, filter=filter)
        except KeyError:
            raise ValueError(
                "Could not find the index"
            )

    def get(self, filter: Dict[str, str]):
        try:
            return self.storage_context.vector_store.get_data(filter=filter)
        except KeyError:
            raise ValueError(
                "Could not find the index"
            )


    def summarization(self, documents: [Document]) -> str:

        chunks_text = ''
        for d in documents:
            chunks_text += d.page_content + '\n'

        if not self.service_context.llm:
            return sentence_transformer_summarize(chunks_text)
        else:
            openai_client=OpenAI()
            model_version="gpt-3.5-turbo"
            query_messages=[]
            instruction_count=1
            if self.service_context.instruction:
                instruction_count+=1

            query_messages.append({"role": "system", "content": f"Follow these {instruction_count} instructions below in all your responses:"})
            if self.service_context.instruction:
                query_messages.append({"role": "system", "content": self.service_context.instruction})
            query_messages.append({"role": "system", "content": f"""Provide concise summary of context below:
            {chunks_text}
            """})

            if self.service_context.version:
                model_version=self.service_context.version
            response = openai_client.chat.completions.create(
                model=model_version,
                messages=query_messages
            )
            if len(response.choices)>0:
                return response.choices[0].message.content
            return None
        '''else:
            if self.service_context.instruction:
                chain = load_summarize_chain(llm=self.service_context.llm, chain_type="map_reduce", verbose=True,
                                             map_prompt=map_prompt_template)
            else:
                chain = load_summarize_chain(llm=self.service_context.llm, chain_type="map_reduce")

        return chain.run(documents)
        '''

    def question_answer(self, query: str, documents: [Document]):

        chunks_text = ''
        for d in documents:
            chunks_text += d.page_content + '\n'
        if not self.service_context.llm:
            return sentence_transformer_question_answer(query,chunks_text)
        else:
            openai_client=OpenAI()
            model_version="gpt-3.5-turbo"
            query_messages=[]
            instruction_count=1
            if self.service_context.instruction:
                instruction_count+=1

            query_messages.append({"role": "system", "content": f"Follow these {instruction_count} instructions below in all your responses:"})
            if self.service_context.instruction:
                query_messages.append({"role": "system", "content": self.service_context.instruction})
            query_messages.append({"role": "system", "content": f"""Answer the question using context below:
            {chunks_text}
            """})
            query_messages.append({"role": "user", "content": query})

            if self.service_context.version:
                model_version=self.service_context.version
            response = openai_client.chat.completions.create(
                model=model_version,
                messages=query_messages
            )
            if len(response.choices)>0:
                return response.choices[0].message.content
            return None

        #old way of using langchain, if we start using LLM other than openai, might need to reuse below code
        '''else :
            if self.service_context.instruction:
                chain = load_qa_chain(llm=self.service_context.llm, chain_type="stuff", verbose=True,
                                      prompt=prompt_template)
            else:
                chain = load_qa_chain(llm=self.service_context.llm, chain_type="stuff")

            return chain.run(input_documents=documents, question=query)
        '''

    def question_answer(self, query: str, documents: [Document]):
        if self.service_context.instruction:
            prompt = f'{self.service_context.instruction}' """
            "{context}"
            Question: {question}:
            """
            prompt_template = PromptTemplate(template=prompt, input_variables=["context", "question"])

        if not self.service_context.llm:
            chunks_text = ''
            for d in documents:
                chunks_text += d.page_content + '\n'
            return sentence_transformer_question_answer(query,chunks_text)
        else:
            if self.service_context.instruction:
                chain = load_qa_chain(llm=self.service_context.llm, chain_type="stuff", verbose=True,
                                      prompt=prompt_template)
            else:
                chain = load_qa_chain(llm=self.service_context.llm, chain_type="stuff")

            return chain.run(input_documents=documents, question=query)
