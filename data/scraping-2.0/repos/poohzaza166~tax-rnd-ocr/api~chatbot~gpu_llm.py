import re
from typing import List, Optional, Mapping, Any, Union
from pydantic.v1 import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from stoping_utils import _SentinelTokenStoppingCriteria
import torch
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

class LLMConfig(BaseModel):
    max_length: int 
    do_sample: bool 
    top_p: float 
    top_k: int 
    temperature: float
    repetition_penalty: float 
    early_stopping: bool 

class GpuLLM(LLM):
    model_name: str
    device: str
    config: LLMConfig
    stop_msgs: Any
    _model : Optional[Any]
    _tokenizer: Optional[Any]
    seed: int
    # __fields_set__ = set()
    # def __init__(self, model_name, device, config: LLMConfig, seed: int, stop_msgs: List[str]=None, **kwargs):
    #     # super().__init__(model_name, **kwargs)
    #     self.model_name = model_name
    #     self.device = device
    #     self.config = config
    #     self.stop_msgs = stop_msgs
    #     self.seed = seed
    #     self._model = None
    #     self._tokenizer = None
    #     self._stop_regex = None
    #     self._initialize()
    # class Config:
    #         validate_assignment = False

    def _initialize(self):
        GpuLLM._model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                            device_map=self.device,
                                                            # use_flash_attention_2=True,
                                                            # torch_dtype=torch.bfloat16,
                                                            # trust_remote_code=True,
                                                            # load_in_8bit_fp32_cpu_offload=True,
                                                            # quantization_config=quantization_config,
                                                            # load_in_4bit=True,
                                                            torch_dtype=torch.float16,
                                                            low_cpu_mem_usage=True,
                                                            )  
        GpuLLM._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        GpuLLM.streaming_callback = TextStreamer(GpuLLM._tokenizer)
        # GpuLLM._tokenizer.pad_token_id = 2
        # GpuLLM._tokenizer.eos_token_id = 2

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        tokenized_prompt = self._tokenizer.encode(
            prompt, 
            return_tensors='pt'
        ).to(self._model.device)
        
        torch.manual_seed(self.seed)
        tokenize = self._tokenizer.encode(prompt, return_tensors='pt').to("cuda")
        stop_token_list = []

        # for stop_msg in stop:
        #     sentinel_tokens_input_ids = self._tokenizer(stop_msg, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        #     stopid = _SentinelTokenStoppingCriteria(stop_msg, tokenizer=self._tokenizer, num_repeats=1)
        #     stop_token_list.append(stopid)
        
        for stop_msg in stop:
            sentinel_token_ids_assistant = self._tokenizer(stop_msg, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
            stop_token_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids_assistant, starting_idx=tokenize.shape[-1], stop_string=stop_msg, tokenizer=self._tokenizer))
      
        if self.stop_msgs != None:
            for stop_msg in self.stop_msgs:
                sentinel_token_ids_assistant = self._tokenizer(stop_msg, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
                stop_token_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids_assistant, starting_idx=tokenize.shape[-1], stop_string=stop_msg, tokenizer=self._tokenizer))
        
        # print(self.config.max_length)
        # print(stop_word)
        stopping_criteria_list = StoppingCriteriaList(stop_token_list)

        output = self._model.generate(
            inputs=tokenized_prompt,
            max_length=self.config.max_length,
            do_sample=self.config.do_sample,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
            stopping_criteria=stopping_criteria_list,
            # early_stopping=self.config.early_stopping,
            streamer=self.streaming_callback
        )
        # print(output)

        response = self._tokenizer.decode(output[0], skip_special_tokens=True)
        replaced_text = self.replace_prompt(response, prompt, '')
        return replaced_text
    
    def replace_prompt(self, text, prompt, replacement):
        # print("yes")
        # print(prompt)
        pattern = re.escape(prompt)
        replaced_text = re.sub(pattern, replacement, text)
        return replaced_text


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "stop": self.stop_msgs,
            "config": self.config
        }
    
    def _build_stop_regex(self, stop_msgs: List[str]) -> str:
        stop_patterns = [re.escape(msg) for msg in stop_msgs]
        return f"({'|'.join(stop_patterns)})"
