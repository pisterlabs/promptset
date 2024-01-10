# -*- coding: utf-8 -*-


from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
import torch
import gc
from pydantic import root_validator
from pathlib import Path
import time


# Set model path
MODEL_PATH_VERSION_1 = ""
MODEL_PATH_VERSION_2 = ""


def load_model(model_path, is_auto, model_config):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            model = AutoModel.from_pretrained(model_path, 
                                            config=model_config,
                                            torch_dtype=torch.float16,
                                            trust_remote_code=True,
                                            device_map=device_map(is_auto)).half()
        else:
            model = AutoModel.from_pretrained(model_path, 
                                            config=model_config,
                                            torch_dtype=torch.float16,
                                            trust_remote_code=True).half().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        pass
    return model, tokenizer


def device_map(is_auto):
    if is_auto:
        return "auto"
    else:
        pass


class ChatglmLLM(LLM):

    max_token: int = 10000
    temperature: float = 0.95
    top_p: float = 0.7
    history_len: int = 10
    version: int = 2
    model_path: str = None
    is_auto: bool = False
    pad_token_id: int = 2
    streaming: bool = False
    history: List[List[str]] = []
    model: Any
    tokenizer: Any
    model_config: Any

    @root_validator()
    def validate_environment(cls, values):
        version = values.get("version")
        if not values.get("model_path"):
            if version == 1:
                values["model_path"] = MODEL_PATH_VERSION_1
            elif version == 2:
                values["model_path"] = MODEL_PATH_VERSION_2
            else:
                raise ValueError(f"Invalid version: {version}")
        model_path = values.get("model_path")
        local_path = Path(f'{model_path}')
        is_auto = values.get("is_auto")
        try:
            values["model_config"] = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
            model, tokenizer = load_model(local_path, is_auto, values["model_config"])
            values["model"], values["tokenizer"] = model, tokenizer
        except Exception as e:
            raise ValueError(f"Some error occured while loading model: {e}")
        return values
    
    @property
    def _llm_type(self):
        return "chatglm"
    
    @property
    def _default_params(self):
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
    
    @property
    def _indentifying_params(self):
        return {**{"model_path":self.model_path}, **self._default_params}
    
    def __clear_torch_cache(self):
        gc.collect()
        if torch.has_cuda:
            CUDA_DEVICE = "cuda:0"
            with torch.cuda.device(CUDA_DEVICE):
                torch.cuda.empty_cache()    
                torch.cuda.ipc_collect()
    
    def _call(self, prompt, stop: Optional[List[str]] = None, 
            run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        streaming = self.streaming
        history = self.history
        
        if streaming:
            for stream_response in self.stream(prompt, stop, run_manager):
                response = stream_response
        else:
            with torch.inference_mode():
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.pad_token_id
                )
            self.__clear_torch_cache()
        
        return response

    @torch.inference_mode()
    def stream(self, prompt, stop: Optional[List[str]] = None, 
            run_manager: Optional[CallbackManagerForLLMRun] = None, 
            ):
        for num, (stream_response, _) in enumerate(self.model.stream_chat(self.tokenizer,
            prompt,
            history=history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.pad_token_id)):

            yield stream_response   

    def __llm_memory(self):
        return self.history    

    def __collect_memory(self):
        self.history += [[prompt, response]]
        # TODO: 
