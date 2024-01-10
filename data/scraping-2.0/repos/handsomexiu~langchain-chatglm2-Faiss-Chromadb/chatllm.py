import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from config import *
# import accelerate
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

init_llm = init_llm
init_embedding_model = init_embedding_model

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):#所有的操作都将在指定的设备上执行
            torch.cuda.empty_cache()# 这个函数用于清空当前CUDA设备上的缓存内存，这可以帮助释放不再使用的GPU内存，以便在需要时可以更好地利用它。
            torch.cuda.ipc_collect()
            # 这个函数用于执行GPU内存IPC（Inter-Process Communication）收集。
            # IPC收集可以帮助回收被释放的GPU内存，以便其他进程或线程可以使用它


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # 这一段可以参考chatglm2-6b的utils.py文件
    #这段代码的目的是根据输入的 GPU 数量和模型层数来自动配置模型的组件分配到不同的 GPU 上。
    # 这种配置可以确保模型的不同部分在多个 GPU 上并行处理，以提高模型的训练和推理性能。
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    device_map = {
        'transformer.word_embeddings': 0,
        'transformer.final_layernorm': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


class ChatLLM(LLM):
    max_token: int = 5000#这里实验还没有设置，到时再看如何设置
    temperature: float = 0.1
    top_p = 0.9
    history = []
    model_type: str = "chatglm2"
    model_name_or_path: str = init_llm,
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
         
        if self.model_type == 'chatglm2':     
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=self.history,
                max_length=self.max_token,
                temperature=self.temperature,
                top_p = self.top_p,
                # 这里可以看 https://github.com/THUDM/ChatGLM2-6B/blob/main/web_demo.py
            )
            torch_gc()
            if stop is not None:
                response = enforce_stop_tokens(response, stop)
            self.history = self.history + [[None, response]]
            # 这里的history没有考虑query,也就是prompt。只考虑了response
        return response


    def load_llm(self,
                   llm_device=DEVICE,
                   num_gpus='auto',
                   device_map: Optional[Dict[str, int]] = None,
                   **kwargs):
        # if 'chatglm2' in self.model_name_or_path.lower():
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                    trust_remote_code=True, cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path))   
                                 
        if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2 and device_map is None:
                self.model = (AutoModel.from_pretrained(
                    self.model_name_or_path, trust_remote_code=True, cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path), 
                    **kwargs).half().cuda())
            else:
                from accelerate import dispatch_model

                model = AutoModel.from_pretrained(self.model_name_or_path,
                                                trust_remote_code=True, cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path),
                                                **kwargs).half()

                if device_map is None:
                    device_map = auto_configure_device_map(num_gpus)

                self.model = dispatch_model(model, device_map=device_map)
        else:#这里就是cpu的了
            self.model = (AutoModel.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True, cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path)).float().to(llm_device))
        self.model = self.model.eval()
