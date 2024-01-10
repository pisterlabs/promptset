from typing import Union
from typing import Optional
from typing import List

import os
import copy
import json
import dotenv
import numpy as np
import datamol as dm

from tqdm.auto import tqdm
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.base import Embeddings as LangChainEmbeddings
from molfeat.trans.pretrained import PretrainedMolTransformer

from molfeat_hype.memory import EmbeddingConversationMemory
from molfeat_hype.utils import convert_smiles
from molfeat_hype.utils import CACHE_DIR

dotenv.load_dotenv()


DEFAULT_SYSTEM_PROMPT = """MolAssistant is a powerful large language model trained by the Open Drug Discovery consortium, 
specifically designed to assist with various tasks related to drug discovery. Its capabilities range from providing 
molecular representation to designing new molecules with specific optimization objectives in mind. Being pre-trained 
on the largest corpus of chemical and biological data, MolAssistant has a deep understanding of chemical structures 
and extensive biological knowledge. It can generate both human-like text and numerical outputs, providing concise and 
accurate responses that are coherent with the topic and instructions given.

MolAssistant is constantly learning and improving, which allows it to process and understand vast amounts of chemical 
and biological data, and it can comprehend any molecular structure. The model's knowledge enables it to provide valuable 
insights and information for a wide range of tasks, including molecular search, QSAR model building, and molecular generation,
making it an indispensable tool for advancing the science of molecular modeling and design in drug discovery.

{history}
Human: {human_input}
MolAssistant:"""

MODEL_EMBEDDING_INSTRUCTIONS = {
    "instructor": "Represent the following molecule for {context}:",
    "openai": """I want you to provide {dimension} dimensional numerical vector at a precision of {precision} as a representation of molecules in the SMILES format that you will receive as input.
    You should first start by understanding the chemical structure and electronic properties of the input molecules before generating the {dimension}-dimensional representation for the following task: {context}.
    To obtain the output, I will provide you with either a single SMILES command or a list of SMILES commands, and you will reply with the most accurate and informative {dimension}-dimensional representation in a json parseable format where the keys are the molecules and the values their representations.
    When generating the output, please ensure that the format is consistent with the task and the instruction given.  Do not write explanations. Do not type anything else unless I instruct you to do so. 
    In case of any invalid or unrecognized SMILES inputs, please provide a suitable error message. My first molecule is c1ccccc1.""",
}


class InstructLLMTransformer(PretrainedMolTransformer):
    """
    Instruction-following Large Language Model Embeddings Transformer for molecules. This transformer embeds molecules using available LLM through langchain.
    Note that the LLMs do not have any molecular context as they were not trained on molecules or any specific molecular task.
    They are just trained on a large corpus of text.
    """

    SUPPORTED_EMBEDDINGS = [
        "hkunlp/instructor-large",
        "hkunlp/instructor-base",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/chatgpt",  # alias for "openai/gpt-3.5-turbo"
    ]

    def __init__(
        self,
        kind=Union[str, LangChainEmbeddings],
        embedding_size: int = 32,
        context: Optional[str] = "modelling",
        standardize: bool = True,
        precompute_cache: bool = True,
        n_jobs: int = 0,
        conv_buffer_size: int = 10,
        conv_max_tokens: Optional[int] = None,
        dtype=float,
        openai_api_key: Optional[str] = None,
        precision: int = 5,
        batch_size: Optional[int] = None,
        system_prompt: Optional[str] = None,
        parallel_kwargs: Optional[dict] = None,
        **params,
    ):
        """Instantiate an instruction following LLM transformer for molecular embeddings

        Args:
            kind: type or name of the model to use for embeddings
            embedding_size: size of the embeddings to return for chat-like models
            context: context to give to the prompt for returning the results. Default is "modelling" which is the context for the modelling instructions.
            standardize: if True, standardize smiles before embedding
            precompute_cache: if True, add a cache to cache the embeddings for the same molecules.
            n_jobs: number of jobs to use for preprocessing smiles.
            conv_buffer_size: conversation buffer size so assistant can remember previous conversations and context for generating features.
            conv_max_tokens: maximum number of tokens to use for the conversation context. If None, will not use a token size limitations.
            dtype: data type to use for the embeddings return type
            openai_api_key: openai api key to use. If None, will try to get it from the environment variable OPENAI_API_KEY
            precision: float precision of the output vector
            batch_size: batch size to use for embedding molecules. If None, will not use a batch size
            system_prompt: system prompt to use for chat-like models. If None, will use the default prompt.
            **params: parameters to pass to the LLM embeddings. See langchain documentation
        """

        self.kind = kind
        self.model = None
        self.standardize = standardize
        self.context = context or "modelling"
        self.embedding_size = embedding_size
        self.precision = precision
        self.batch_size = batch_size
        self.conv_max_tokens = conv_max_tokens
        self.conv_buffer_size = conv_buffer_size
        self._length = None
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        params.setdefault("temperature", 0.8)
        if isinstance(kind, str):
            if not (
                kind.startswith(tuple(self.SUPPORTED_EMBEDDINGS)) or kind.startswith("openai/")
            ):
                raise ValueError(
                    f"Unknown LLM type {kind} requested. Supported models are {self.SUPPORTED_EMBEDDINGS}"
                )
            if kind.startswith("openai/"):
                if kind == "openai/chatgpt":
                    kind = "openai/gpt-3.5-turbo"
                if openai_api_key is None:
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                prompt = PromptTemplate(
                    input_variables=["history", "human_input"], template=self.system_prompt
                )
                llm = ChatOpenAI(
                    model_name=kind.replace("openai/", ""),
                    openai_api_key=openai_api_key,
                    **params,
                )
                self.model = LLMChain(
                    llm=llm,
                    prompt=prompt,
                    verbose=False,
                    memory=EmbeddingConversationMemory(
                        k=self.conv_buffer_size,
                        ai_prefix="MolAssistant",
                        llm=llm,
                        max_token_limit=self.conv_max_tokens,
                    ),
                )
                output = self.model.predict(
                    human_input=MODEL_EMBEDDING_INSTRUCTIONS["openai"].format(
                        dimension=embedding_size, precision=self.precision, context=self.context
                    )
                )
                embeddings = json.loads(output)
                if "c1ccccc1" not in embeddings:
                    raise ValueError(
                        "Model is not able to understand the prompt. Please select a different model"
                    )
                assert (
                    len(embeddings["c1ccccc1"]) == embedding_size
                ), "Model cannot return the correct embedding size."
                self._length = embedding_size

            elif kind.startswith("llama.cpp") or kind.startswith("gpt4all"):
                raise ValueError(f"{kind} is not yet supported, because or how slow they are.")
            else:
                # we need to remove temperature key
                params.pop("temperature", None)
                self.model = HuggingFaceInstructEmbeddings(
                    model_name=kind,
                    embed_instruction=MODEL_EMBEDDING_INSTRUCTIONS["instructor"].format(
                        context=self.context
                    ),
                    model_kwargs=params,
                    cache_folder=CACHE_DIR,
                )
        super().__init__(
            precompute_cache=precompute_cache,
            n_jobs=n_jobs,
            dtype=dtype,
            device="cpu",
            parallel_kwargs=parallel_kwargs,
            **params,
        )

    def __len__(self):
        """Get the length of the featurizer"""
        if self._length is not None:
            return self._length
        return super().__len__()

    def _convert(self, inputs: List[Union[str, dm.Mol]], **kwargs):
        """Convert the list of input molecules into the proper format for embeddings"""
        self._preload()
        parallel_kwargs = copy.deepcopy(getattr(self, "parallel_kwargs", {}))
        parallel_kwargs["n_jobs"] = self.n_jobs
        return convert_smiles(inputs, parallel_kwargs, standardize=self.standardize)

    def _embed(self, smiles: List[str], **kwargs):
        """_embed takes a list of smiles or molecules and return the featurization
        corresponding to the inputs.  In `transform` and `_transform`, this function is
        called after calling `_convert`

        Args:
            smiles: input smiles
        """
        if isinstance(self.model, LangChainEmbeddings):
            return self.model.embed_documents(smiles)
        # basically running embeddings
        # compute expected total token for inputs based on expected number of char
        expected_tokens = (self.embedding_size * (self.precision + 5) + 4) * len(smiles) + sum(
            len(x) + 5 for x in smiles
        )
        # we splits the number of smiles to avoid being over the maximum tokens
        if not self.batch_size:
            maximum_tokens = self.model.llm.max_tokens or self.model.memory.max_token_limit or 2000
            n_splits = max(1, int(np.ceil(expected_tokens / maximum_tokens)))
        else:
            n_splits = max(1, int(np.ceil(len(smiles) / self.batch_size)))
        data = {}
        for batch in tqdm(np.array_split(smiles, n_splits), desc=f"Batch embedding", leave=False):
            json_output = self.model.predict(human_input=" ,".join(batch))
            batch_data = json.loads(json_output)
            # EN: surprisingly, ChatGPT can return randomized version of a SMILES
            data.update({dm.unique_id(k.strip()): v for k, v in batch_data.items()})
        missed_embedding = np.full_like(list(data.values())[0], np.nan)
        data = [data.get(dm.unique_id(sm), missed_embedding) for sm in smiles]
        return data
