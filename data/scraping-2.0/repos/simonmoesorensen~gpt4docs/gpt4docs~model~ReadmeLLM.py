from pathlib import Path
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

TOKENS_LIMIT = {
    "gpt-3.5-turbo": 4096,
    "text-davinci-003": 4096,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo-16k": 16384,
}

prompt_dir = Path(__file__).parent / "prompts"


class ReadmeLLM:
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
        self.reduce_llm = ChatOpenAI(model_name="gpt-4", streaming=False, temperature=0)
        self.retriever = retriever

        readme_prompt = PromptTemplate.from_template(
            open(prompt_dir / "readme.txt").read()
        )
        map_prompt = PromptTemplate.from_template(open(prompt_dir / "map.txt").read())

        self.chain = load_summarize_chain(
            llm=self.model,
            reduce_llm=self.reduce_llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=readme_prompt,
            input_key="input_documents",
            output_key="output_text",
            return_intermediate_steps=True,
        )

    def run(self) -> str:
        docs = self.get_relevant_docs()
        response = self.chain({"input_documents": docs})
        logger.debug("\n\n".join(response["intermediate_steps"]))
        return response["output_text"]

    async def arun(self) -> str:
        docs = self.get_relevant_docs()
        response = await self.chain.acall({"input_documents": docs})
        logger.debug("\n\n".join(response["intermediate_steps"]))
        return response["output_text"]

    def get_relevant_docs(self):
        return self.retriever._get_relevant_documents(
            query="How do I get started with the project?",
            run_manager=None,
        )
