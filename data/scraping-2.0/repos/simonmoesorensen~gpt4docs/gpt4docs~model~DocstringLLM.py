from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from gpt4docs.modules.datamodels import PyDefinition

TOKENS_LIMIT = {
    "gpt-3.5-turbo": 4096,
    "text-davinci-003": 4096,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo-16k": 16384,
}

prompt_dir = Path(__file__).parent / "prompts"


class DocstringLLM:
    def __init__(
        self,
        retriever=None,
        callbacks=None,
        model_name="gpt-3.5-turbo-16k",
    ):
        """
        Setup the langchain Chain class for Q&A with LLM
        """
        if model_name not in TOKENS_LIMIT:
            raise ValueError(
                f"Model {model_name} not supported. "
                f"Supported models: {TOKENS_LIMIT.keys()}"
            )

        if callbacks is None:
            callbacks = []

        self.callbacks = callbacks
        self.model = ChatOpenAI(model_name=model_name, streaming=False, temperature=0)
        self.retriever = retriever

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    open(prompt_dir / "qa.txt").read()
                ),
                HumanMessagePromptTemplate.from_template(
                    "Write a docstring for the following definition: `{question}`\nGenerated Docstring:"  # noqa: E501
                ),
            ]
        )

        # Setup final chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": qa_prompt,
            },
            retriever=self.retriever,
        )

    def run(self, definition: PyDefinition) -> str:
        response = self.chain.run(definition.source)
        return self._format_response(response)

    async def arun(self, definition: PyDefinition) -> str:
        response = await self.chain.arun(query=definition.source)
        return self._format_response(response)

    def _format_response(self, response: str) -> str:
        return response.replace('"""', "")
