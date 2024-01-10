from __future__ import annotations
import os

from typing import Any, Dict, List, Optional
from langchain import LLMChain
from pydantic import Extra
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

apiKey = os.getenv("OPENAI_API_KEY")


from pydantic import BaseModel, Field


class KeyQuestions(BaseModel):
    """The Key Questions."""

    questions: List[str] = Field(..., description="the keyQuestions")


class contextChain(Chain):
    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    iterations: int = 1
    output_key: str = "text"  #: :meta private:
    context: str = ""
    db: Chroma = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs).to_string()

        self.db = Chroma(
            persist_directory="./langchainPages/db/chroma_db",
            embedding_function=OpenAIEmbeddings(openai_api_key=apiKey),
        )

        for i in range(self.iterations):
            keyQuestions = self.getKeyQuestions(prompt_value, run_manager=run_manager)

            for question in keyQuestions.questions:
                self.queryDatabaseAndAddToContext(question)

        response = self.getAnswerWithContext(prompt_value, run_manager=run_manager)

        return {self.output_key: response}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response}

    def getKeyQuestions(
        self,
        prompt_value: str,
        run_manager: Optional[CallbackManagerForChainRun],
    ):
        prompt_msgs = []
        input = prompt_value
        prompt_msgs.append(
            SystemMessage(
                content="You are a System that can extract the key questions, that need to be answered for a detailed answer to a given question."
            )
        )
        if self.context != "":
            prompt_msgs.append(
                SystemMessage(
                    content="You have a context of already given information so the key questions should ask for missing information."
                )
            )
            prompt_msgs.append(
                HumanMessagePromptTemplate.from_template(
                    "This is your context: {context}"
                )
            )
            input = {"input": prompt_value, "context": self.context}

        prompt_msgs.append(
            HumanMessagePromptTemplate.from_template(
                "Extract the 3 key question of following question: {input}"
            )
        )

        prompt = ChatPromptTemplate(messages=prompt_msgs)
        chain = create_structured_output_chain(
            output_schema=KeyQuestions,
            llm=self.llm,
            prompt=prompt,
            verbose=True,
        )

        keyQuestions: KeyQuestions = chain.run(
            input, callbacks=run_manager.get_child() if run_manager else None
        )

        return keyQuestions

    def queryDatabaseAndAddToContext(self, input: str, answers=2):
        docs = self.db.similarity_search(input, k=answers)
        for doc in docs:
            if doc.page_content not in self.context:
                self.context += input + "\n" + doc.page_content + "\n\n"

    def getAnswerWithContext(
        self, prompt_value: str, run_manager: Optional[CallbackManagerForChainRun]
    ):
        prompt_msgs = [
            SystemMessage(
                content="You are a System that can answer questions based on a given context."
            ),
            HumanMessagePromptTemplate.from_template("This is your context: {context}"),
            HumanMessagePromptTemplate.from_template(
                "Now answer the following question: {input}"
            ),
        ]

        prompt = ChatPromptTemplate(messages=prompt_msgs)

        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)

        return chain.run(
            {"input": prompt_value, "context": self.context},
            callbacks=run_manager.get_child() if run_manager else None,
        )

    @property
    def _chain_type(self) -> str:
        return "context_chain"
