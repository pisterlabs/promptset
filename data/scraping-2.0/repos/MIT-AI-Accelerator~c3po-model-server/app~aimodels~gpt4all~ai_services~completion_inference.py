from contextlib import redirect_stdout
import io
import os
from time import time
from uuid import uuid4
from enum import Enum
from typing import Optional
from pydantic import BaseModel, validator, ValidationError
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pathlib import Path
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllModelFilenameEnum
from app.core.minio import download_file_from_minio
from minio import Minio
from app.core.model_cache import MODEL_CACHE_BASEDIR
from app.core.logging import logger, LogConfig
from logging.config import dictConfig
from ..models import Gpt4AllPretrainedModel

dictConfig(LogConfig().dict())

BASE_CKPT_DIR = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "./data")

class InitInputs(BaseModel):
    gpt4all_pretrained_model_obj: Gpt4AllPretrainedModel
    s3: Minio

    # ensure that model type is defined
    @validator('gpt4all_pretrained_model_obj')
    def gpt4all_pretrained_model_obj_must_have_model_type_and_be_uploaded(cls, v):
        # pylint: disable=no-self-argument
        if not v.model_type:
            raise ValueError(
                'gpt4all_pretrained_model_obj must have model_type')
        if not v.uploaded:
            raise ValueError(
                'gpt4all_pretrained_model_obj must be uploaded')

        return v

    class Config:
        arbitrary_types_allowed = True


class FinishReasonEnum(str, Enum):
    """
    Possible Reasons for how the Completion finished (same as OpenAI)
        stop: API returned complete model output
        length: Incomplete model output due to max_tokens parameter or token limit
        content_filter: Omitted content due to a flag from our content filters
        null: API response still in progress or incomplete

    """
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    NULL = "null"

class CompletionInferenceOutputChoices(BaseModel):
    """
    Choices object for models output by the OpenAI SDK.  Example below:
    {
        "text": "\n\nThis is indeed a test",
        "index": 0,
        "logprobs": null,
        "finish_reason": "length"
    }
    """

    text: str
    index: int = 0
    logprobs: Optional[list[float]] = None
    finish_reason: FinishReasonEnum = FinishReasonEnum.LENGTH

class CompletionInferenceOutputUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionInferenceOutputs(BaseModel):
    """
    Models the outputs expected by the OpenAI SDK.  Example below:
    {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [
            {
            "text": "\n\nThis is indeed a test",
            "index": 0,
            "logprobs": null,
            "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12
        }
    }

    Note: Right now usage is not implemented, only types are stubbed out in code.
    """
    id: str = str(uuid4())
    object: str = "text_completion"
    created: int = int(time())
    model: Optional[Gpt4AllModelFilenameEnum] = None
    choices: list[CompletionInferenceOutputChoices]
    usage: Optional[CompletionInferenceOutputUsage] = None

    class Config:
        arbitrary_types_allowed = True

class CompletionInferenceInputs(BaseModel):
    """
    Basic pydantic verification class for collecting inputs that map
    to the OpenAI SDK.  Unless otherwise stated,
    the names are the same between both Gpt4All and OpenAI.
    Currently, the following items from the OpenAI API are not supported:
    - suffix
    - logprobs
    - frequency_penalty
    - best_of
    - logit_bias
    - user
    """

    model: Optional[Gpt4AllModelFilenameEnum] = None
    """Name of the GPT4All model file.
    NOTE 1: While this does map to model on Gpt4All, the meaning is slightly different.
    Here it is the name of the model; there it is the full model path.
    NOTE 2: This is defined here for use in the API, however, by the time the
    CompletionInference object is initialized, this is already set"""

    prompt: str
    """The prompt for the completions."""

    max_tokens: Optional[int] = 256
    """The maximum number of tokens to generate.
    NOTE: maps to n_predict on Gpt4All"""

    temperature: Optional[float] = 0.8
    """The temperature to use for sampling.
    NOTE: maps to temp on Gpt4All"""

    top_p: Optional[float] = 0.95
    """The top-p value to use for sampling. Uses
    nucleus sampling, where the model considers the results
    of the tokens with top_p probability mass. So 0.1 means
    only the tokens comprising the top 10% probability mass
    are considered.
    """

    n: int = 1
    """Number of times to replicate this prompt and generate a unique
    output.
    NOTE: does not map to anything on Gpt4All, implemented via
    LangChain generate method."""

    stream: bool = False
    """Whether to stream the results or not.
    NOTE: maps to streaming on Gpt4All, however we allow LangChain
    to handle this via callbacks."""

    echo: Optional[bool] = False
    """Whether to echo the prompt."""

    stop: Optional[list[str]] = []
    """A list of strings to stop generation when encountered."""

    presence_penalty: Optional[float] = 1.3
    """The penalty to apply to repeated tokens.
    NOTE: maps to repeat_penalty on Gpt4All"""

    class Config:
        arbitrary_types_allowed = True

class CompletionInference:

    def __init__(self, gpt4all_pretrained_model_obj, s3):

        # validate input
        InitInputs(
            gpt4all_pretrained_model_obj=gpt4all_pretrained_model_obj, s3=s3
        )

        self.gpt4all_pretrained_model_obj = gpt4all_pretrained_model_obj
        self.llm_path = os.path.join(MODEL_CACHE_BASEDIR, gpt4all_pretrained_model_obj.model_type)

        if not os.path.isfile(self.llm_path):
            # Create the directory if it doesn't exist
            Path(self.llm_path).parent.mkdir(parents=True, exist_ok=True)

            # filename or id
            minio_filename = gpt4all_pretrained_model_obj.model_type if gpt4all_pretrained_model_obj.use_base_model else gpt4all_pretrained_model_obj.id

            # Download the file from Minio
            logger.info(f"Downloading model from Minio to {self.llm_path}")
            download_file_from_minio(minio_filename, s3, filename=self.llm_path)
            logger.info(f"Downloaded model from Minio to {self.llm_path}")


    def basic_response(self, api_inputs: CompletionInferenceInputs):
        return self._general_completion_base(api_inputs)

    def chat_response(self, api_inputs: CompletionInferenceInputs):
        template = """Prompt: {api_prompt}

        Response: \n\n"""
        return self._general_completion_base(api_inputs, template=template)

    def question_response(self, api_inputs: CompletionInferenceInputs):
        template = """Question: {api_prompt}

        Answer: Let's think step by step."""

        return self._general_completion_base(api_inputs, template=template)

    def _general_completion_base(self,
                                 api_inputs: CompletionInferenceInputs,
                                 template: str = """{api_prompt}""",
                                 ):

        # validate input
        if not isinstance(api_inputs, CompletionInferenceInputs):
            raise ValidationError(
                'must input type BasicResponseInputs')

        # build the prompt template
        prompt = PromptTemplate(template=template, input_variables=["api_prompt"])

        # build the chain
        llm = self._build_llm(api_inputs)
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # run inference
        api_input_list = [{ "api_prompt": api_inputs.prompt } for item in range(api_inputs.n)]

        # use context manager to redirect stream output so multiple requests
        # can be handled at once on the same process (else output streams conflict)
        # https://stackoverflow.com/questions/1218933/can-i-redirect-the-stdout-into-some-sort-of-string-buffer
        with io.StringIO() as buf, redirect_stdout(buf):
            results = llm_chain.generate(api_input_list)

        choices = []
        for generation in results.generations:
            finish_reason = FinishReasonEnum.NULL
            if (generation[0].generation_info) and ("finish_reason" in generation[0].generation_info):
                finish_reason = generation[0].generation_info['finish_reason']

            choices.append(CompletionInferenceOutputChoices(
                text=generation[0].text,
                index=0,
                finish_reason=finish_reason
            ))

        return CompletionInferenceOutputs(
            model=self.gpt4all_pretrained_model_obj.model_type,
            choices=choices
        )

    def _build_llm(self,
                   api_inputs: CompletionInferenceInputs
                   ):

        # handle stream via LangChain Callbacks
        # Callbacks support token-wise streaming
        callbacks = []
        if (api_inputs.stream):
            callbacks.append(StreamingStdOutCallbackHandler())

        # Verbose is required to pass to the callback manager
        # TODO If you want to use a custom model add the backend parameter (e.g., backend='gptj'),
        # see https://docs.gpt4all.io/gpt4all_python.html
        llm = GPT4All(
            model=self.llm_path,
            callbacks=callbacks,
            verbose=False,
            n_predict=api_inputs.max_tokens,
            temp=api_inputs.temperature,
            top_p=api_inputs.top_p,
            echo=api_inputs.echo,
            stop=api_inputs.stop,
            repeat_penalty=api_inputs.presence_penalty,
            use_mlock = True
        )

        return llm
