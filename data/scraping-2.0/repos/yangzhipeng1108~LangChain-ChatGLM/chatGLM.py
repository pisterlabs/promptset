from typing import List, Optional
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer
from config import Config
import lora_utils.insert_lora
import torch

class LLMService(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "LLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response]]
        return response

    def load_model(self, model_name_or_path: str = "THUDM/chatglm-6b"):
        """
        加载大模型LLM
        :return:
        """
        device_map = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.llm_model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map).half()
        self.model = self.model.eval()
        
     def load_model_with_lora(self, model_name_or_path: str = "THUDM/chatglm-6b"):
        """
        加载大模型LLM
        :return:
        """
        device_map = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.llm_model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map).half()
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

        # 加载基于belle  110万数据微调的lora权重
        peft_path = Config.lora_path

        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM, inference_mode=False,
        #     r=8,
        #     lora_alpha=32, lora_dropout=0.1
        # )
        # model = get_peft_model(model, peft_config)

        lora_config = {
            'r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'enable_lora': [True, True, True],
        }
        self.model = lora_utils.insert_lora.get_lora_model(self.model, lora_config)
        self.model.load_state_dict(torch.load(peft_path), strict=False)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        self.model = self.model.eval()

if __name__ == '__main__':
    chatLLM = LLMService()
    chatLLM.load_model()
