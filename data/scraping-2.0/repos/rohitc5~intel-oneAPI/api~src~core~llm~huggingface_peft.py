"""Wrapper around HuggingFace Peft APIs."""
import importlib.util
import logging
from typing import Any, List, Mapping, Optional

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "huggyllama/llama-7b"
DEFAULT_ADAPTER_ID = "timdettmers/qlora-flan-7b"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text-generation")

logger = logging.getLogger(__name__)


class HuggingFacePEFT(LLM):
    """Wrapper around HuggingFace Pipeline API.

    To use, you should have the ``transformers` and `peft`` python packages installed.

    Only supports `text-generation` for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain.llms import HuggingFacePipeline
            hf = HuggingFacePipeline.from_model_id(
                model_id="gpt2",
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 10},
            )
    """

    model: Any  #: :meta private:
    tokenizer: Any
    
    device: int = -1
    """Device to use"""
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    adapter_id: str = DEFAULT_ADAPTER_ID
    """Adapter name to use"""
    model_kwargs: Optional[dict] = None
    """Key word arguments passed to the model."""
    generation_kwargs: Optional[dict] = None
    """Generation arguments passed to the model."""
    quantization_kwargs: Optional[dict] = None
    """Quantization arguments passed to the quantization."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        adapter_id: str,
        task: str,
        device: int = -1,
        model_kwargs: Optional[dict] = {},
        generation_kwargs: Optional[dict] = {},
        quantization_kwargs: Optional[dict] = {}
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM, 
                AutoModelForSeq2SeqLM,
                AutoTokenizer, 
                BitsAndBytesConfig
            )
            from peft import PeftModel    
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers peft`."
            )

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            quantization_config = None
            if quantization_kwargs:
                bnb_4bit_compute_dtype = quantization_kwargs.get("bnb_4bit_compute_dtype", torch.float32)
                if bnb_4bit_compute_dtype == "bfloat16":
                    quantization_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
                elif bnb_4bit_compute_dtype == "float16":
                    quantization_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                elif bnb_4bit_compute_dtype == "float32":
                    quantization_kwargs["bnb_4bit_compute_dtype"] = torch.float32
                
                quantization_config = BitsAndBytesConfig(**quantization_kwargs)
            
            torch_dtype = _model_kwargs.get("torch_dtype", torch.float32)
            if torch_dtype is not None:
                if torch_dtype == "bfloat16":
                    _model_kwargs["torch_dtype"] = torch.bfloat16
                elif torch_dtype == "float16":
                    _model_kwargs["torch_dtype"] = torch.float16
                elif torch_dtype == "float32":
                    _model_kwargs["torch_dtype"] = torch.float32
            
            
            max_memory= {i: _model_kwargs.get(
                "max_memory", "24000MB") for i in range(torch.cuda.device_count())}
            _model_kwargs.pop("max_memory")
            if task == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    max_memory=max_memory,
                    quantization_config=quantization_config,
                    **_model_kwargs
                )
                model = PeftModel.from_pretrained(model, adapter_id)
            else:
                raise ValueError(
                    f"Got invalid task {task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
        except ImportError as e:
            raise ValueError(
                f"Could not load the {task} model due to missing dependencies."
            ) from e

        if importlib.util.find_spec("torch") is not None:

            cuda_device_count = torch.cuda.device_count()
            if device < -1 or (device >= cuda_device_count):
                raise ValueError(
                    f"Got device=={device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
            if device < 0 and cuda_device_count > 0:
                logger.warning(
                    "Device has %d GPUs available. "
                    "Provide device={deviceId} to `from_model_id` to use available"
                    "GPUs for execution. deviceId is -1 (default) for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )
        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_kwargs=model_kwargs,
            generation_kwargs=generation_kwargs,
            quantization_kwargs=quantization_kwargs
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "generation_kwargs": self.generation_kwargs,
            "quantization_kwargs": self.quantization_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_peft"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        import torch
        from transformers import GenerationConfig
        from transformers import StoppingCriteria, StoppingCriteriaList

        device_id = "cpu"
        if self.device != -1:
            device_id = "cuda:{}".format(self.device)

        stopping_criteria = None
        stop_sequence = self.generation_kwargs.get("stop_sequence", [])
        if len(stop_sequence) > 0:
            stop_token_ids = [self.tokenizer(
                stop_word, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_sequence]
            stop_token_ids = [token.to(device_id) for token in stop_token_ids]

            # define custom stopping criteria object
            class StopOnTokens(StoppingCriteria):
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    for stop_ids in stop_token_ids:
                        if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                            return True
                    return False
            stopping_criteria = StoppingCriteriaList([StopOnTokens()])

            # eos_token_id = stop_token_ids[0][0].item()
            # self.generation_kwargs["pad_token_id"] = 1
            # self.generation_kwargs["eos_token_id"] = eos_token_id
        
        self.generation_kwargs.pop("stop_sequence")
        generation_config = GenerationConfig(**self.generation_kwargs)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device_id)
        outputs = self.model.generate(
            inputs=inputs.input_ids, 
            stopping_criteria=stopping_criteria,
            generation_config=generation_config
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = text[len(prompt) :]
        if len(stop_sequence) > 0:
            text = enforce_stop_tokens(text, stop_sequence)

        return text