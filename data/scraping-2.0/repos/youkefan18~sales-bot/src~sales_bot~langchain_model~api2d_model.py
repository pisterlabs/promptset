from typing import (
    AbstractSet,
    Any,
    Collection,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Field, root_validator
from langchain.utils import get_from_dict_or_env
from utils import LOG


class Api2dLLM(LLM):
    # Instead of extending BaseOpenAI, subclassing LLM makes it easy to customize "_call"
    # For BaseOpenAI, a proper client is needed and an override of __call__ or _generate might be needed
    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    deployment_name: str = "gpt-3.5-turbo"
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""
    presence_penalty: float = 0
    """Penalizes repeated tokens."""
    n: int = 1
    """How many completions to generate for each prompt."""
    best_of: int = 1
    """Generates best_of completions server-side and returns the "best"."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://oa.api2d.net"
    openai_api_completion: str = "/v1/completions"
    openai_api_chatcompletion: str = "/v1/chat/completions"
    openai_organization: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """Adjust the probability of specific tokens being generated."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    """Set of special tokens that are allowed。"""
    disallowed_special: Union[Literal["all"], Collection[str]] = "all"
    """Set of special tokens that are not allowed。"""
    tiktoken_model_name: Optional[str] = None
    stop: Optional[List[str]] = None
    class Config():
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(values, "openai_api_key", "API2D_OPENAI_API_KEY")
        values["openai_api_base"] = get_from_dict_or_env(values, "openai_api_base", "API2D_OPENAI_API_BASE")
        values["openai_api_completion"] = get_from_dict_or_env(values, "openai_api_completion", "API2D_OPENAI_API_COMPLETION")
        values["openai_api_chatcompletion"] = get_from_dict_or_env(values, "openai_api_chatcompletion", "API2D_OPENAI_API_CHAT_COMPLETION")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling AI21 API."""
        return {
            "openai_api_base": self.openai_api_base,
            "openai_api_completion": self.openai_api_completion,
            "openai_api_chatcompletion": self.openai_api_chatcompletion,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "request_timeout": self.request_timeout,
            "logit_bias": self.logit_bias,
        }

    # @property
    # def _invocation_params(self) -> Dict[str, Any]:
    #     openai_params = {
    #         "engine": self.deployment_name
    #     }
    #     return {**openai_params, **super()._invocation_params}

    @property
    def _llm_type(self) -> str:
        return "api2d"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        params = {**self._default_params, **kwargs}
        #completion json
        is_gpt3_5: bool = self.model_name == "gpt-3.5-turbo"
        input = {"prompt": prompt, "stopSequences": stop, **params}
        url = f"{self.openai_api_base}/{self.openai_api_completion}"
        if is_gpt3_5:
            input = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            url = f"{self.openai_api_base}/{self.openai_api_chatcompletion}"

        response = requests.post(
            url=url,
            headers={"Authorization": f"Bearer {self.openai_api_key}"},
            json=input,
        )
        if response.status_code != 200:
            optional_detail = response.json().get("error")
            raise ValueError(
                f"Api2d call failed with status code {response.status_code}."
                f" Details: {optional_detail}"
            )
        
        #return response.json()["completions"][0]["data"]["text"]
        return response.json()["choices"][0]["message"]["content"].strip() if is_gpt3_5 \
            else response.json()["choices"][0]["text"].strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model_name}, **self._default_params}
    
if __name__ == "__main__":
    import os

    import dotenv
    env_file = f'{os.getenv("ENVIRONMENT", "dev")}.env'
    dotenv.load_dotenv(dotenv_path=f'{env_file}')

    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    llm = Api2dLLM()
    prompt = PromptTemplate(
        input_variables=["product_desc"],
        template="给一个{product_desc}的系统取10个符合特性，吸引眼球的产品名字，给出完整名称",
    )
    #llm("which year is this year?")
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run({
        'product_desc': "能通过多种文件访问协议访问如s3，NFS,Samba访问不同生物实验室设备数据的，方便用户访问并分享实验室文档及图片系统的，含有相关数据dashboard的"
        }))