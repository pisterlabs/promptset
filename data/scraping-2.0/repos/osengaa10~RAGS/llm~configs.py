import os
import together
import textwrap
from typing import Any, Dict
from pydantic import Extra
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationKGMemory


load_dotenv()
os.environ['TOGETHER_API_KEY']
together.api_key = os.environ["TOGETHER_API_KEY"]


class TogetherLLM(LLM):
    """Together large language models."""
    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""
    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""
    class Config:
        extra = Extra.forbid
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"
    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text


# # Instantiate embeddings model
# model_name = "BAAI/bge-base-en"
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# model_norm = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs=encode_kwargs
# )


model_name = "BAAI/bge-large-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

embedding = model_norm

llm = TogetherLLM(
    model= "mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature = 0.1,
    max_tokens = 4000,
)