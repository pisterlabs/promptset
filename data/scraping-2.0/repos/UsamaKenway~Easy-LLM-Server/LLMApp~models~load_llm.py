from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, \
    BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from transformers import StoppingCriteriaList
from LLMApp.utils.stop_sequence import StopOnTokens
import torch
import gc
import time


class HFModel:
    def __init__(self, model_name: str) -> None:
        self.llm = None
        self.pipe = None
        self.base_model = None
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        stop_token_ids = self.tokenizer('</s>', add_special_tokens=False,
                                        return_tensors="pt").input_ids
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    def set_hf_pipe(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.base_model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            # max_length=256,
            temperature=0.6,
            top_p=0.95,
            # top_k=0,
            repetition_penalty=1.1,
            stopping_criteria=self.stopping_criteria,
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def load_model(self):
        # self.base_model = AutoGPTQForCausalLM.from_quantized(self.model_name,
        #                                                      device="cuda:0",
        #                                                      use_safetensors=True)
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                               quantization_config=nf4_config,
                                                               use_auth_token=False)
        # self.base_model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     quantization_config=nf4_config,
        #     device_map="auto",
        #     # load_in_4bit=True,
        #     use_auth_token=True
        # )

        self.set_hf_pipe()
        # return self.llm

    def update_stop_sequence(self, human_predix):
        stop_token_text = f"\n{human_predix}"
        stop_token_ids = self.tokenizer(stop_token_text, add_special_tokens=False,
                                        return_tensors="pt").input_ids[:, 1:]
        print(stop_token_ids)
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        self.set_hf_pipe()
        return self.llm


class GPTQModel(HFModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_model(self):
        self.base_model = AutoGPTQForCausalLM.from_quantized(self.model_name,
                                                             device="cuda:0",
                                                             use_safetensors=True)
        self.set_hf_pipe()
        return self.llm
