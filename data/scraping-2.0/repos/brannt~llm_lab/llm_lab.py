# TODO: Replace prints with logging
# TODO: Pretty-print results with tabulate
import importlib
import itertools
import time
from pathlib import Path

import pydantic
from langchain import LLMChain, PromptTemplate
from langchain.llms.base import LLM

LLM_PROVIDERS = {
    "gpt4all": {"class": "GPT4All", "model_param": "model", "model_type": "local"},
    "llamacpp": {
        "class": "LlamaCpp",
        "model_param": "model_path",
        "model_type": "local",
    },
    "huggingface": {
        "class": "HuggingFaceHub",
        "model_param": "repo_id",
        "model_type": "hosted",
    },
    "openai": {"class": "OpenAI", "model_param": None, "model_type": "hosted"},
    "replicate": {"class": "Replicate", "model_param": "model", "model_type": "hosted"},
    "cerebrium": {
        "class": "CerebriumAI",
        "model_param": "endpoint_url",
        "model_type": "hosted",
    },
}
LLMS = [
    {
        "provider": "replicate",
        "model": "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    },
    {"provider": "openai"},
    {"provider": "gpt4all", "model": "ggml-gpt4all-j-v1.3-groovy.bin"},
    {"provider": "gpt4all", "model": "ggml-model-gpt4all-falcon-q4_0.bin"},
    {"provider": "llamacpp", "model": "wizardlm-7b-uncensored-ggml-q4_0.bin"},
]
VARIABLES = {
    "prompt_template": [
        {"template": "Q: {question} A: ", "input_variables": ["question"]},
        {
            "template": "Question: {question} Answer: Let's think step by step.",
            "input_variables": ["question"],
        },
    ],
    "prompt_inputs": [
        {"question": "How many 50x50 cm floor tiles are needed for a 70m2 flat?"},
        {"question": "What is the capital of France?"},
    ],
}

local_dir = Path("/mnt/c/Models/")


class LLMProviderParams(pydantic.BaseModel):
    class_name: str
    model_param: str
    model_type: str = "local"


class LLMParams(pydantic.BaseModel):
    provider: str
    model: str = None


llm_cache: dict[LLMParams, LLM]


def get_llm(llm_params: LLMParams):
    global llm_cache
    if llm_params in llm_cache:
        return llm_cache[llm_params]
    # Clean cache from unused local LLMs to avoid memory crashes
    for params in llm_cache:
        if LLM_PROVIDERS[params.provider]["model_type"] == "local":
            del llm_cache[params]
    llm = load_llm(llm_params)
    llm_cache[llm_params] = llm
    return llm


def load_llm(llm_params: LLMParams):
    provider_params = LLM_PROVIDERS[llm_params.provider]
    llm_class = provider_params["class"]
    if "." in llm_class:
        llm_module, llm_class = llm_class.rsplit(".", 1)
    else:
        llm_module = "langchain.llms"
    try:
        llm_class = getattr(importlib.import_module(llm_module), llm_class)
    except (ImportError, AttributeError):
        raise ValueError(
            f"Could not load LLM class {llm_class} from module {llm_module}"
        )
    if llm_params.model is None:
        return llm_class()
    if provider_params["model_type"] == "local":
        return llm_class(
            **{provider_params["model_param"]: str(local_dir / llm_params.model)}
        )
    else:
        return llm_class(**{provider_params["model_param"]: llm_params.model})


class BaseLLMRunner(pydantic.BaseModel):
    llm_params: LLMParams

    @property
    def llm(self):
        return load_llm(self.llm_params)

    def run(self):
        raise NotImplementedError


class SimpleLLMRunner(BaseLLMRunner):
    prompt: str

    def run(self):
        return self.llm(self.prompt)


class LLMChainRunner(BaseLLMRunner):
    prompt_template: PromptTemplate
    prompt_inputs: dict = None

    def run(self):
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        return chain.run(self.prompt_inputs or {})


def run_lab(runner_class, llms=LLMS, variables=VARIABLES):
    results = []
    runner_class = LLMChainRunner
    for llm_params in llms:
        for vars in itertools.product(*variables.values()):
            vars = dict(zip(variables.keys(), vars))
            runner = runner_class(llm_params=llm_params, **vars)
            print(runner)
            print(f"Testing {llm_params} with {vars}")
            time_start = time.time()
            try:
                output = runner.run()
            except Exception as e:
                output = str(e)
                raise
            inference_time = time.time() - time_start
            print(output)
            results.append([llm_params, vars, inference_time, output])

    return results
