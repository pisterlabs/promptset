from typing import Dict, List, Any, Optional, Callable
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Extra
from langchain.schema.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch
import numpy
import warnings, os
from akasha.models.llama2 import Llama2, TaiwanLLaMaGPTQ


class chatGLM(LLM):
    max_token: int = 4096
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any
    model: Any

    def __init__(self, model_name: str, temperature: float = 0.01):
        """define chatglm model and the tokenizer

        Args:
            **model_name (str)**: chatglm model name\n
        """
        if model_name == "":
            model_name = "THUDM/chatglm2-6b"

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, device="cuda"
        )
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.01

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        self.model = self.model.eval()
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response


class custom_embed(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any  #: :meta private:
    model_name: str = "custom embedding model"
    """Model name to use."""
    # model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = {}
    """Keyword arguments to pass when calling the `encode` method of the model."""

    def __init__(self, func: Any, encode_kwargs: Dict[str, Any] = {}, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.client = func
        self.encode_kwargs = encode_kwargs

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))

        embeddings = self.client(texts, **self.encode_kwargs)

        if isinstance(embeddings, numpy.ndarray):
            embeddings = embeddings.tolist()

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]


class custom_model(LLM):
    max_token: int = 4096
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any
    model: Any
    func: Any

    def __init__(self, func: Callable, temperature: float = 0.001):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        self.func = func
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.001

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return self.func.__name__

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """

        response = self.func(prompt)
        return response


def get_hf_model(model_name, temperature: float = 0.0):
    """try different methods to define huggingface model, first use pipline and then use llama2.

    Args:
        model_name (str): huggingface model name\n

    Returns:
        _type_: llm model
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        try:
            if hf_token is None:
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={
                        "temperature": temperature,
                        "repetition_penalty": 1.2,
                    },
                    device_map="auto",
                    max_new_tokens=512,
                    batch_size=1,
                    torch_dtype=torch.float16,
                )
                model = HuggingFacePipeline(pipeline=pipe)
            else:
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    use_auth_token=hf_token,
                    max_new_tokens=512,
                    model_kwargs={
                        "temperature": temperature,
                        "repetition_penalty": 1.2,
                    },
                    device_map="auto",
                    batch_size=1,
                    torch_dtype=torch.float16,
                )
                model = HuggingFacePipeline(pipeline=pipe)
        except:
            if model_name.lower().find("taiwan-llama") != -1:
                model = TaiwanLLaMaGPTQ(
                    model_name_or_path=model_name, temperature=temperature
                )
            else:
                model = Llama2(
                    model_name_or_path=model_name,
                    temperature=temperature,
                    bit4=True,
                    max_token=4096,
                )

    return model
