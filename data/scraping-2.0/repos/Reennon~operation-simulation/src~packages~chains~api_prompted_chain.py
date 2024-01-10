from typing import Dict, Optional, Sequence, Any, cast, List

from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.chains import APIChain, LLMChain
from langchain.chains.api.base import _check_in_allowed_domain
from langchain.chains.api.prompt import API_URL_PROMPT, API_RESPONSE_PROMPT
from langchain.chains.base import Chain
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.utilities import TextRequestsWrapper
from pydantic import root_validator, BaseModel, validator
from pydantic.fields import FieldInfo, Field
from pydantic_core import ArgsKwargs

from src.packages.constants.config_constants import ConfigConstants


class APIPromptedChain(Chain):
    """Chain that makes API calls and summarizes the responses to answer a question.

    *Security Note*: This API chain uses the requests toolkit
        to make GET, POST, PATCH, PUT, and DELETE requests to an API.

        Exercise care in who is allowed to use this chain. If exposing
        to end users, consider that users will be able to make arbitrary
        requests on behalf of the server hosting the code. For example,
        users could ask the server to make a request to a private API
        that is only accessible from the server.

        Control access to who can submit issue requests using this toolkit and
        what network access it has.

        See https://python.langchain.com/docs/security for more information.
    """

    api_request_chain: LLMChain
    api_answer_chain: LLMChain
    requests_wrapper: TextRequestsWrapper = Field(exclude=True)
    api_docs: str
    question_key: str = "query"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    limit_to_domains: Optional[Sequence[str]] = None
    """Use to limit the domains that can be accessed by the API chain.

    * For example, to limit to just the domain `https://www.example.com`, set
        `limit_to_domains=["https://www.example.com"]`.

    * The default value is an empty tuple, which means that no domains are
      allowed by default. By design this will raise an error on instantiation.
    * Use a None if you want to allow all domains by default -- this is not
      recommended for security reasons, as it would allow malicious users to
      make requests to arbitrary URLS including internal APIs accessible from
      the server.
    """

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    @root_validator(pre=True)
    def validate_limit_to_domains(cls, values: Dict) -> Dict:
        """Check that allowed domains are valid."""
        if "limit_to_domains" not in values:
            raise ValueError(
                "You must specify a list of domains to limit access using "
                "`limit_to_domains`"
            )
        if not values["limit_to_domains"] and values["limit_to_domains"] is not None:
            raise ValueError(
                "Please provide a list of domains to limit access using "
                "`limit_to_domains`."
            )
        return values

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.question_key]
        api_url = self.api_request_chain.predict(
            query=question,
            api_docs=self.api_docs,
            callbacks=_run_manager.get_child(),
        )
        _run_manager.on_text(api_url, color="green", end="\n", verbose=self.verbose)
        api_url = api_url.strip()
        if self.limit_to_domains and not _check_in_allowed_domain(
                api_url, self.limit_to_domains
        ):
            raise ValueError(
                f"{api_url} is not in the allowed domains: {self.limit_to_domains}"
            )
        api_response = self.requests_wrapper.get(api_url)
        _run_manager.on_text(
            api_response, color="yellow", end="\n", verbose=self.verbose
        )
        answer = self.api_answer_chain.predict(
            query=question,
            api_docs=self.api_docs,
            api_url=api_url,
            api_response=api_response,
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: answer}

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.question_key]
        api_url = await self.api_request_chain.apredict(
            question=question,
            api_docs=self.api_docs,
            callbacks=_run_manager.get_child(),
        )
        await _run_manager.on_text(
            api_url, color="green", end="\n", verbose=self.verbose
        )
        api_url = api_url.strip()
        if self.limit_to_domains and not _check_in_allowed_domain(
                api_url, self.limit_to_domains
        ):
            raise ValueError(
                f"{api_url} is not in the allowed domains: {self.limit_to_domains}"
            )
        api_response = await self.requests_wrapper.aget(api_url)
        await _run_manager.on_text(
            api_response, color="yellow", end="\n", verbose=self.verbose
        )
        answer = await self.api_answer_chain.apredict(
            question=question,
            api_docs=self.api_docs,
            api_url=api_url,
            api_response=api_response,
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: answer}

    @classmethod
    def from_llm_and_api_docs(
            cls,
            llm: BaseLanguageModel,
            api_docs: str,
            headers: Optional[dict] = None,
            api_url_prompt: BasePromptTemplate = API_URL_PROMPT,
            api_response_prompt: BasePromptTemplate = API_RESPONSE_PROMPT,
            limit_to_domains: Optional[Sequence[str]] = None,
            **kwargs: Any,
    ) -> APIChain:
        """Load chain from just an LLM and the api docs."""
        get_request_chain = LLMChain(llm=llm, prompt=api_url_prompt)
        requests_wrapper = TextRequestsWrapper(headers=headers)
        get_answer_chain = LLMChain(llm=llm, prompt=api_response_prompt)
        return cls(
            api_request_chain=get_request_chain,
            api_answer_chain=get_answer_chain,
            requests_wrapper=requests_wrapper,
            api_docs=api_docs,
            limit_to_domains=limit_to_domains,
            **kwargs,
        )

    @property
    def _chain_type(self) -> str:
        return "api_chain"