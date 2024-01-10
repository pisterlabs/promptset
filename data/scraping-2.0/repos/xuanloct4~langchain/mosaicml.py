#!pip install apify-client

import environment
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Set
from pydantic import Extra, Field, root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights
from tqdm.auto import tqdm
from threading import Thread
from huggingface_hub import snapshot_download, cached_assets_path

"""Wrapper for the MosaicML MPT models."""
class MosaicML(LLM):
    model_name: str = Field("mosaicml/mpt-7b-storywriter", alias='model_name')
    """The name of the model to use."""

    tokenizer_name: str = Field("EleutherAI/gpt-neox-20b", alias='tokenizer_name')
    """The name of the sentence tokenizer to use."""

    config: Any = None #: :meta private:
    """The reference to the loaded configuration."""

    tokenizer: Any = None #: :meta private:
    """The reference to the loaded tokenizer."""

    model: Any = None #: :meta private:
    """The reference to the loaded model."""

    accelerator: Any = None #: :meta private:
    """The reference to the loaded hf device accelerator."""

    attn_impl: str = Field("torch", alias='attn_impl')
    """The attention implementation to use."""

    torch_dtype: Any = Field(torch.bfloat16, alias='torch_dtype')
    """The torch data type to use."""

    max_new_tokens: Optional[int] = Field(10000, alias='max_new_tokens')
    """The maximum number of tokens to generate."""

    do_sample: Optional[bool] = Field(True, alias='do_sample')
    """Whether to sample or not."""

    temperature: Optional[float] = Field(0.8, alias='temperature')
    """The temperature to use for sampling."""

    echo: Optional[bool] = Field(False, alias='echo')
    """Whether to echo the prompt."""
    
    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid


    def _mpt_default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }
    
    @staticmethod
    def _mpt_param_names() -> Set[str]:
        """Get the identifying parameters."""
        return {
            "max_new_tokens",
            "temperature",
            "do_sample",
        }

    @staticmethod
    def _model_param_names(model_name: str) -> Set[str]:
        """Get the identifying parameters."""
        # TODO: fork for different parameters for different model variants.
        return MosaicML._mpt_param_names()
    
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return self._mpt_default_params()
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the environment."""
        try:
            # This module is supermassive so we use the transformers accelerator to load it.
            values['accelerator'] = Accelerator()
            print("[" + values["model_name"] + "] Downloading model (or fetching from cache)...")
            download_location = snapshot_download(repo_id=values["model_name"], use_auth_token=False, local_files_only=False)
            print("[" + values["model_name"] + "] Model location: " + str(download_location))
            offload_cache_location = cached_assets_path(library_name="langchain", namespace=values["model_name"], subfolder="offload")
            print("[" + values["model_name"] + "] Offload cache location: " + str(offload_cache_location))
            print("[" + values["model_name"] + "] AutoConfiguring...")
            values["config"] = AutoConfig.from_pretrained(values["model_name"], trust_remote_code=True)
            values["config"].attn_config['attn_impl'] = values["attn_impl"]
            values["tokenizer"] = AutoTokenizer.from_pretrained(values["tokenizer_name"])
            print("[" + values["model_name"] + "] Initializing empty weights for model...")
            with init_empty_weights():
                values["model"] = AutoModelForCausalLM.from_pretrained(
                    values["model_name"],
                    config=values["config"],
                    torch_dtype=values["torch_dtype"],
                    trust_remote_code=True
                )
            print("[" + values["model_name"] + "] Tying weights...")
            values["model"].tie_weights()
            print("[" + values["model_name"] + "] Dispatching checkpoint...")
            values["model"] = load_checkpoint_and_dispatch(
                values["model"], 
                download_location, 
                device_map="auto", 
                no_split_module_classes=["MPTBlock"],
                offload_folder=offload_cache_location
            )
            print("[" + values["model_name"] + "] Loaded successfully!")
        except Exception as e:
            raise Exception(f"MosaicML failed to load with error: {e}")
        return values
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model_name,
            **self._default_params(),
            **{
                k: v
                for k, v in self.__dict__.items()
                if k in self._model_param_names(self.model_name)
            },
        }
    
    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "mosaicml"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        r"""Call out to MosiacML's generate method via transformers.

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "This is a story about a big sabre tooth tiger: "
                response = model(prompt)
        """
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""
        inputs = self.tokenizer([prompt], return_tensors='pt')
        inputs = inputs.to(self.accelerator.device)
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(inputs, streamer=streamer, **self._mpt_default_params())
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        text = ""
        pbar = tqdm(total=self.max_new_tokens, desc="Thinking", leave=False)
        for new_text in streamer:
            if text_callback:
                text_callback(new_text)
            text += new_text
            pbar.update(1)
        pbar.close()
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text

llm = MosaicML(model_name='mosaicml/mpt-7b-storywriter', attn_impl='torch', torch_dtype=torch.bfloat16, max_new_tokens=200, echo=True)

llm("Tell me a short story about sabretooth tigers.")
