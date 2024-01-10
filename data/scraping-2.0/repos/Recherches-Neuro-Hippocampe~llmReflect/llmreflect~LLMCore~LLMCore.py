from typing import Any, List, Dict, Union, Optional, Generator
from langchain.chains import LLMChain
from llmreflect.Prompt.BasicPrompt import BasicPrompt
from langchain.base_language import BaseLanguageModel
from abc import ABC
from langchain.chat_models import ChatOpenAI
from llmreflect.Utils.log import get_logger
from llmreflect.Utils.log import check_current_openai_balance
from llmreflect.Utils.log import general_trace_var
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.manager import CallbackManagerForChainRun
import inspect
from langchain.load.dump import dumpd
from langchain.schema import RUN_KEY, RunInfo, LLMResult
from langchain.llms import LlamaCpp
import os
import logging
from pydantic import Field
from langchain.callbacks.manager import CallbackManagerForLLMRun
import gc

from decouple import config
logger = logging.getLogger(__name__)

Callbacks = Optional[Union[List[BaseCallbackHandler], BaseCallbackManager]]


class LOCAL_MODEL:
    """
    A constant class storing local model path
    """
    base_dir = config("LOCAL_MODEL")
    upstage_70_b = os.path.join(
        base_dir,
        "upstage-llama-2-70b-instruct-v2.ggmlv3.q5_K_M.bin")
    upstage_30_b = os.path.join(
        base_dir,
        "upstage-llama-30b-instruct-2048.ggmlv3.q8_0.bin")
    llama2_uncensor_70_b = os.path.join(
        base_dir,
        "llama2_70b_chat_uncensored.ggmlv3.q5_K_S.bin"
    )
    guanaco_70b_b = os.path.join(
        base_dir,
        'llama-2-70b-guanaco-qlora.ggmlv3.q5_K_S.bin'
    )


class OPENAI_MODEL:
    """
    A constant class storing openai models
    """
    gpt_4 = "gpt-4"
    gpt_4_0314 = "gpt-4-0314"
    gpt_4_0613 = "gpt-4-0613"
    gpt_4_32k = "gpt-4-32k"
    gpt_4_32k_0314 = "gpt-4-32k-0314"
    gpt_4_32k_0613 = "gpt-4-32k-0613"
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_3_5_turbo_0301 = "gpt-3.5-turbo-0301"
    gpt_3_5_turbo_0613 = "gpt-3.5-turbo-0613"
    gpt_3_5_turbo_16k = "gpt-3.5-turbo-16k"
    gpt_3_5_turbo_16k_0613 = "gpt-3.5-turbo-16k-0613"
    text_ada_001 = "text-ada-001"
    ada = "ada"
    text_babbage_001 = "text-babbage-001"
    babbage = "babbage"
    text_curie_001 = "text-curie-001"
    curie = "curie"
    davinci = "davinci"
    text_davinci_003 = "text-davinci-003"
    text_davinci_002 = "text-davinci-002"
    code_davinci_002 = "code-davinci-002"
    code_davinci_001 = "code-davinci-001"
    code_cushman_002 = "code-cushman-002"
    code_cushman_001 = "code-cushman-001"


class LLMCore(LLMChain, ABC):
    def __init__(self, prompt: BasicPrompt, llm: BaseLanguageModel):
        super().__init__(prompt=prompt.get_langchain_prompt_template(),
                         llm=llm)
        """
        Abstract class for core functions of LLM,
        inherit from the LLM chain class.
        Args:
            prompt (BasicPrompt): Prompt class to use.
            llm_model (BaseLanguageModel): llm class to use
        """
        object.__setattr__(self, "logger", get_logger(self.__class__.__name__))
        object.__setattr__(self, "max_output_tokens", self.llm.max_tokens)
        object.__setattr__(self, "model_name", "LLM")

    def get_inputs(self) -> List[str]:
        """
        showing inputs for the prompt template being used
        Returns:
            List: a list of strings
        """
        return self.prompt.input_variables

    @property
    def is_local(self) -> bool:
        """
        Tell whether a model is local model or openai model.
        Returns:
            bool: If local model then true, otherwise false.
        """
        if self.model_name in OPENAI_MODEL.__dict__.values():
            return False
        return True


def singleton(cls):
    """A decorator wrapper.
    Used to wrap Llama2Cpp class.
    Make sure there is only one instance running at the time.
    Otherwise the GPU V memory will be drained.

    Returns:
        Nah.
    """
    instances = []

    def get_instance(*args, **kwargs):
        if len(instances) == 0:
            report_gpu()
            instances.append(cls(*args, **kwargs))
        return instances[0]
    return get_instance


def in_workflow():
    return os.getenv("GITHUB_ACTIONS")\
        or os.getenv("TRAVIS") \
        or os.getenv("CIRCLECI") \
        or os.getenv("GITLAB_CI")


def report_gpu():
    """Check gpu usage and also clean the cache on gpu.
    """
    if not bool(in_workflow):
        import torch
        print(torch.cuda.list_gpu_processes())
        gc.collect()
        torch.cuda.empty_cache()
    else:
        pass


@singleton
class Llama2Cpp(LlamaCpp):
    """Wrapper around the llama.cpp model.

    To use, you should have the llama-cpp-python library installed,
    and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out: https://github.com/abetlen/llama-cpp-python

    Example:
        .. code-block:: python
            llm = Llama2Cpp(model_path="/path/to/llama/model")
    """

    client: Any  #: :meta private:
    model_path: str
    """The path to the Llama model file."""

    lora_base: Optional[str] = None
    """The path to the Llama LoRA base model."""

    lora_path: Optional[str] = None
    """The path to the Llama LoRA. If None, no LoRa is loaded."""

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into.
    If -1, the number of parts is automatically determined."""

    seed: int = Field(-1, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(True, alias="f16_kv")
    """Use half-precision for key/value cache."""

    logits_all: bool = Field(False, alias="logits_all")
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = Field(False, alias="vocab_only")
    """Only load the vocabulary, no weights."""

    use_mlock: bool = Field(False, alias="use_mlock")
    """Force system to keep model in RAM."""

    n_threads: Optional[int] = Field(None, alias="n_threads")
    """Number of threads to use.
    If None, the number of threads is automatically determined."""

    n_batch: Optional[int] = Field(8, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    n_gpu_layers: Optional[int] = Field(None, alias="n_gpu_layers")
    """Number of layers to be loaded into gpu memory. Default None."""

    suffix: Optional[str] = Field(None)
    """A suffix to append to the generated text.
    If None, no suffix is appended."""

    max_tokens: Optional[int] = 256
    """The maximum number of tokens to generate."""

    temperature: Optional[float] = 0.8
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.95
    """The top-p value to use for sampling."""

    logprobs: Optional[int] = Field(None)
    """The number of logprobs to return. If None, no logprobs are returned."""

    echo: Optional[bool] = False
    """Whether to echo the prompt."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    repeat_penalty: Optional[float] = 1.1
    """The penalty to apply to repeated tokens."""

    top_k: Optional[int] = 40
    """The top-k value to use for sampling."""

    last_n_tokens_size: Optional[int] = 64
    """The number of tokens to look back when applying the repeat_penalty."""

    use_mmap: Optional[bool] = True
    """Whether to keep the model loaded in RAM"""

    streaming: bool = True
    """Whether to stream the results, token by token."""

    verbose: bool = True
    """Print verbose output to stderr."""
    n_gqa: int = 8
    """Mandatory settings for 65b model"""

    # @model_validator(mode="after")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        model_path = values["model_path"]
        model_param_names = [
            "lora_path",
            "lora_base",
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
            "use_mmap",
            "last_n_tokens_size",
            "verbose",
            "n_gqa"
        ]
        model_params = {k: values[k] for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if values["n_gpu_layers"] is not None:
            model_params["n_gpu_layers"] = values["n_gpu_layers"]

        try:
            from llama_cpp import Llama

            values["client"] = Llama(model_path, **model_params)
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import llama-cpp-python library. "
                "Please install the llama-cpp-python library to "
                "use this embedding model: pip install llama-cpp-python"
            )
        except Exception as e:
            raise ValueError(
                f"Could not load Llama model from path: {model_path}. "
                f"Received error {e}"
            )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling llama_cpp."""
        return {
            "suffix": self.suffix,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": self.logprobs,
            "echo": self.echo,
            "stop_sequences": self.stop,
            # key here is convention among LLM classes
            "repeat_penalty": self.repeat_penalty,
            "top_k": self.top_k,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_path": self.model_path}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llamacpp"

    def _get_parameters(self,
                        stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Performs sanity check, preparing parameters
        in format needed by llama_cpp.

        Args:
            stop (Optional[List[str]]): List of stop sequences for llama_cpp.

        Returns:
            Dictionary containing the combined parameters.
        """

        # Raise error if stop sequences are in both input and default params
        if self.stop and stop is not None:
            raise ValueError(
                "`stop` found in both the input and default params.")

        params = self._default_params

        # llama_cpp expects the "stop" key not this, so we remove it:
        params.pop("stop_sequences")

        # then sets it as configured, or default to an empty list:
        params["stop"] = self.stop or stop or []

        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Llama model and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                from langchain.llms import LlamaCpp
                llm = LlamaCpp(model_path="/path/to/local/llama/model.bin")
                llm("This is a prompt.")
        """
        if self.streaming:
            # If streaming is enabled, we use the stream
            # method that yields as they are generated
            # and return the combined strings from the first choices's text:
            combined_text_output = ""
            for token in self.stream(prompt=prompt,
                                     stop=stop,
                                     run_manager=run_manager):
                combined_text_output += token["choices"][0]["text"]
            return combined_text_output
        else:
            params = self._get_parameters(stop)
            params = {**params, **kwargs}
            result = self.client(prompt=prompt, **params)
            return result["choices"][0]["text"]

    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Generator[Dict, None, None]:
        """Yields results objects as they are generated in real time.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens being generated.

        Yields:
            A dictionary like objects containing a string token and metadata.
            See llama-cpp-python docs and below for more.
        """
        params = self._get_parameters(stop)
        result = self.client(prompt=prompt, stream=True, **params)
        for chunk in result:
            token = chunk["choices"][0]["text"]
            log_probs = chunk["choices"][0].get("logprobs", None)
            if run_manager:
                run_manager.on_llm_new_token(
                    token=token, verbose=self.verbose, log_probs=log_probs
                )
            yield chunk

    def get_num_tokens(self, text: str) -> int:
        tokenized_text = self.client.tokenize(text.encode("utf-8"))
        return len(tokenized_text)


class OpenAICore(LLMCore):
    def __init__(self, open_ai_key: str,
                 prompt_name: str = '',
                 max_output_tokens: int = 512,
                 temperature: float = 0.0,
                 llm_model=OPENAI_MODEL.gpt_3_5_turbo):
        """
        LLMCore class specifically designed for openAI.
        Args:
            open_ai_key (str): OpenAI key
            prompt_name (str, optional): name for the prompt. Defaults to ''.
            max_output_tokens (int, optional): maximum number of output tokens.
                Defaults to 512.
            temperature (float, optional): Flexibility of the output.
                Defaults to 0.0.
            llm_model (str, optional): string indicating the mode to use.
                Should be included in class LLM_BACKBONE_MODEL.
                Defaults to LLM_BACKBONE_MODEL.gpt_3_5_turbo.
        """
        prompt = BasicPrompt.\
            load_prompt_from_json_file(prompt_name)
        llm = ChatOpenAI(temperature=temperature,
                         openai_api_key=open_ai_key,
                         model=llm_model)
        llm.max_tokens = max_output_tokens
        super().__init__(prompt=prompt,
                         llm=llm)
        object.__setattr__(self, "model_name", self.llm.model_name)

    def get_inputs(self) -> List[str]:
        """
        showing inputs for the prompt template being used
        Returns:
            List: A list of input variable, each one should be str
        """
        return self.prompt.input_variables

    def predict(self, **kwargs: Any) -> str:
        """
        The llm prediction interface.
        Returns:
            str: The output / completion generated by llm.
        """
        return self._predict(inputs=kwargs,
                             callbacks=[general_trace_var.get()])

    def _predict(
        self,
        inputs: Union[Dict[str, Any], Any],
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
    ) -> Dict[str, Any]:
        """Execute the chain.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param. Should contain all inputs specified in
                `Chain.input_keys` except for inputs that will be set by the
                chain's memory.
            return_only_outputs: Whether to return only outputs in the
                response. If True, only new keys generated by this chain will
                be returned. If False, both input keys and new keys generated
                by this chain will be returned. Defaults to False.
            callbacks: Callbacks to use for this chain run. These will be
                called in addition to callbacks passed to the chain during
                construction, but only these runtime callbacks will propagate
                to calls to other objects.
            tags: List of string tags to pass to all callbacks. These will be
                passed in addition to tags passed to the chain during
                construction, but only these runtime tags will propagate to
                calls to other objects.
            metadata: Optional metadata associated with the chain.
                Defaults to None
            include_run_info: Whether to include run info in the response.
                Defaults to False.

        Returns:
            A dict of named outputs. Should contain all outputs specified in
                `Chain.output_keys`.
        """
        inputs = self.prep_inputs(inputs)
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._call).\
            parameters.get("run_manager")
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            inputs,
        )
        try:
            outputs = (
                self._call(inputs, run_manager=run_manager)
                if new_arg_supported
                else self._call(inputs)
            )
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise e
        run_manager.on_chain_end(outputs)
        final_outputs: Dict[str, Any] = self.prep_outputs(
            inputs, outputs, return_only_outputs
        )
        if include_run_info:
            final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return final_outputs[self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """call function used in _predict function

        Args:
            inputs (Dict[str, Any]): inputs prepared by `self.prep_input`
            run_manager (Optional[CallbackManagerForChainRun], optional):
                run manager provided by callback manager. Defaults to None.

        Returns:
            Dict[str, str]: llm outputs
        """
        response = self.generate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """
        The core function for generating LLM result from inputs.
        By using the "check_current_openai_balance". The generation
        will be stopped when the cost is going to exceed the budget.
        """
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        run_permit = check_current_openai_balance(
            input_prompt=prompts[0].to_string(),
            max_output_tokens=self.max_output_tokens,
            model_name=self.model_name,
            logger=self.logger)
        if run_permit:
            return self.llm.generate_prompt(
                prompts,
                stop,
                callbacks=run_manager.get_child() if run_manager else None,
                **self.llm_kwargs,
            )
        else:
            raise Exception("Budget Error: The next round text completion \
is likely to exceed the budget. LLM is forced to stop.")


class LlamacppCore(LLMCore):
    def __init__(self,
                 model_path: str,
                 prompt_name: str = '',
                 max_total_tokens: int = 2048,
                 max_output_tokens: int = 512,
                 temperature: float = 0.0,
                 verbose: bool = False,
                 n_gpus_layers: int = 8,
                 n_threads: int = 16,
                 n_batch: int = 512
                 ):
        """
        The LLMCore class for Llamacpp.
        Args:
            model_path (str): Path to the model.
            prompt_name (str, optional): name for prompt. Defaults to ''.
            max_total_tokens (int, optional): Maximum context size.
                Defaults to 2048.
            max_output_tokens (int, optional): Maximum size of completion.
                Defaults to 512.
            temperature (float, optional): Flexibility of the model.
                Defaults to 0.0.
            verbose (bool, optional): whether to print the status.
                Defaults to False.
            n_gpus_layers (int, optional): number of layer to load on gpu.
                Defaults to 8.
            n_threads (int, optional): Number of threads to use.
                Defaults to 16.
            n_batch (int, optional): Maximum number of prompt tokens to batch
                together when calling llama_eval.
                Defaults to 512.
        """
        prompt = BasicPrompt.\
            load_prompt_from_json_file(prompt_name)
        llm = Llama2Cpp(
            model_path=model_path,
            n_ctx=max_total_tokens,
            max_tokens=max_output_tokens,
            temperature=temperature,
            n_gpu_layers=n_gpus_layers,
            n_threads=n_threads,
            n_batch=n_batch,
            verbose=verbose
        )

        super().__init__(prompt=prompt,
                         llm=llm)
        model_name = os.path.basename(self.llm.model_path)
        object.__setattr__(self, "model_name", model_name)

    def get_inputs(self) -> List[str]:
        """
        showing inputs for the prompt template being used
        Returns:
            List: A list of input variable, each one should be str
        """
        return self.prompt.input_variables

    def predict(self, **kwargs: Any) -> str:
        """
        The llm prediction interface.
        Returns:
            str: The output / completion generated by llm.
        """
        predicted_result = self._predict(inputs=kwargs,
                                         callbacks=[general_trace_var.get()])
        return predicted_result

    def _predict(
        self,
        inputs: Union[Dict[str, Any], Any],
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
    ) -> Dict[str, Any]:
        """Execute the chain.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param. Should contain all inputs specified in
                `Chain.input_keys` except for inputs that will be set by the
                chain's memory.
            return_only_outputs: Whether to return only outputs in the
                response. If True, only new keys generated by this chain will
                be returned. If False, both input keys and new keys generated
                by this chain will be returned. Defaults to False.
            callbacks: Callbacks to use for this chain run. These will be
                called in addition to callbacks passed to the chain during
                construction, but only these runtime callbacks will propagate
                to calls to other objects.
            tags: List of string tags to pass to all callbacks. These will be
                passed in addition to tags passed to the chain during
                construction, but only these runtime tags will propagate to
                calls to other objects.
            metadata: Optional metadata associated with the chain.
                Defaults to None
            include_run_info: Whether to include run info in the response.
                Defaults to False.

        Returns:
            A dict of named outputs. Should contain all outputs specified in
                `Chain.output_keys`.
        """
        inputs = self.prep_inputs(inputs)
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._call).\
            parameters.get("run_manager")
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            inputs,
        )
        try:
            outputs = (
                self._call(inputs, run_manager=run_manager)
                if new_arg_supported
                else self._call(inputs)
            )
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise e

        run_manager.on_chain_end(outputs)
        final_outputs: Dict[str, Any] = self.prep_outputs(
            inputs, outputs, return_only_outputs
        )
        if include_run_info:
            final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return final_outputs[self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """call function used in _predict function

        Args:
            inputs (Dict[str, Any]): inputs prepared by `self.prep_input`
            run_manager (Optional[CallbackManagerForChainRun], optional):
                run manager provided by callback manager. Defaults to None.

        Returns:
            Dict[str, str]: llm outputs
        """
        response = self.generate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """
        The core function for generating LLM result from inputs.
        By using the "check_current_openai_balance". The generation
        will be stopped when the cost is going to exceed the budget.
        """
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        llmresult = self.llm.generate_prompt(
            prompts,
            stop,
            callbacks=run_manager.get_child() if run_manager else None,
            **self.llm_kwargs,
        )

        prompt_tokens = 0
        for prompt in prompts:
            prompt_tokens += self.llm.get_num_tokens(str(prompt))
        completion_tokens = 0
        for completion in llmresult.generations:
            for candidate in completion:
                completion_tokens += self.llm.get_num_tokens(candidate.text)
        llm_output_info = {
            "token_usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": completion_tokens + prompt_tokens
            },
            "model_name": os.path.basename(self.llm.model_path)
        }

        llmresult.__setattr__("llm_output", llm_output_info)
        for handler in run_manager.handlers:
            handler.on_llm_end(response=llmresult)

        return llmresult
