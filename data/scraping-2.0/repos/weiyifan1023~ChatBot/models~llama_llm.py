from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel, GenerationConfig, LlamaForCausalLM,LlamaTokenizer,AutoModelForCausalLM
import torch
from configs.model_config import LLM_DEVICE

DEVICE = LLM_DEVICE
DEVICE_ID = "0,1,2" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class Llama(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Llama"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        generate_kwargs = {
        "max_new_tokens": 100,
        "min_new_tokens": 1,
        "temperature": 0.1,
        "do_sample": False,  # The three options below used together leads to contrastive search
        "top_k": 1,
        "penalty_alpha": 0.6,
        # "no_repeat_ngram_size": no_repeat_ngram_size,
        # **generation_config,
    }
        #print("***********prompt************")
        #print(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_len = input_ids.shape[1]
        input_ids = input_ids.to(0)
        generate_ids = self.model.generate(input_ids, **generate_kwargs)
        response = self.tokenizer.batch_decode(generate_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #print("********response***************")
        #print(response)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history+[[None, response]]
        return response

    def load_model(self, model_name_or_path: str = "decapoda-research/llama-13b-hf", llm_device=LLM_DEVICE):
        # print device map
        # config = AutoConfig.from_pretrained(model_name_or_path)
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(config)
        #
        # device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"])
        # print(device_map)

        # tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        # model
        if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
            hf_device_map = self.model.hf_device_map
            print(hf_device_map)
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(model_name_or_path).float().to(llm_device)
            )
        self.model = self.model.eval()
        return self.tokenizer, self.model  # wyf add
