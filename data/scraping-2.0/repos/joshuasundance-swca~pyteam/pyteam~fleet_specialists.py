from __future__ import annotations

from typing import Optional

import pandas as pd
from langchain.agents import AgentType, initialize_agent
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema.document import Document
from langchain.schema.runnable import Runnable

from pyteam.fleet_retrievers import MultiVectorFleetRetriever


class FleetBackedSpecialist:
    library_name: str
    retriever: MultiVectorFleetRetriever
    # prompt: ChatPromptTemplate
    llm: BaseLLM
    memory: ConversationBufferMemory

    qa_chain: RetrievalQA
    specialist: Runnable

    # _system_message_template = (
    #     "You are a great software engineer who is very familiar with Python. "
    #     "Given a user question or request about a new Python library "
    #     "called `{library}` and parts of the `{library}` documentation, "
    #     "answer the question or generate the requested code. "
    #     "Your answers must be accurate, should include code whenever possible, "
    #     "and should not assume anything about `{library}` which is not "
    #     "explicitly stated in the `{library}` documentation. "
    #     "If the required information is not available, just say so.\n\n"
    #     "`{library}` Documentation\n"
    #     "------------------\n\n"
    #     "{context}"
    # )
    #
    # _prompt_template = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", _system_message_template),
    #         ("human", "{question}"),
    #     ],
    # )

    @staticmethod
    def _join_docs(docs: list[Document], sep: str = "\n\n") -> str:
        return sep.join(d.page_content for d in docs)

    def __init__(
        self,
        library_name: str,
        retriever: MultiVectorFleetRetriever,
        llm: BaseLLM,
        memory: Optional[ConversationBufferMemory] = None,
    ):
        self.memory = memory or ConversationBufferMemory()
        self.llm = llm
        self.retriever = retriever
        # self.prompt = self._prompt_template.partial(
        #     library=library_name,
        # )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
        )
        self.specialist = initialize_agent(
            [
                Tool(
                    name=f"{library_name} QA System",
                    func=self.qa_chain.run,
                    description=f"Useful for when you need to answer questions about "
                    f"the {library_name} library. Input should be a fully formed question.",
                ),
            ],
            llm,
            agent_kwargs={
                "extra_prompt_messages": [
                    MessagesPlaceholder(variable_name="memory"),
                ],
            },
            memory=self.memory,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        library_name: str,
        llm: BaseLLM,
        **kwargs,
    ) -> FleetBackedSpecialist:
        retriever = MultiVectorFleetRetriever.from_df(
            df,
            library_name,
            **kwargs,
        )
        return cls(library_name, retriever, llm)

    @classmethod
    def from_library(
        cls,
        library_name: str,
        llm: BaseLLM,
        download_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> FleetBackedSpecialist:
        retriever = MultiVectorFleetRetriever.from_library(
            library_name,
            download_kwargs,
            **kwargs,
        )
        return cls(library_name, retriever, llm)

    @classmethod
    def from_parquet(
        cls,
        parquet_path,
        llm: BaseLLM,
        **kwargs,
    ) -> FleetBackedSpecialist:
        retriever = MultiVectorFleetRetriever.from_parquet(
            parquet_path,
            **kwargs,
        )
        library_name = MultiVectorFleetRetriever.get_library_name_from_filename(
            parquet_path,
        )
        return cls(library_name, retriever, llm)
