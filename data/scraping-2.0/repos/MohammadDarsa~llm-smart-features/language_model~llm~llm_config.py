import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline
from langchain.memory import VectorStoreRetrieverMemory
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline


class LlmConfig:

    def __init__(self, vector_db_config):
        self.model_name_or_path = "TheBloke/Dolphin-Llama-13B-GPTQ"
        self.model_basename = "model"
        self.local_llm = None
        self.memory = None
        self.vector_db_config = vector_db_config
        self.config()

    def config(self):
        # go for a smaller model if you don't have the VRAM
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoGPTQForCausalLM.from_quantized(self.model_name_or_path,
                                                   model_basename=self.model_basename,
                                                   use_safetensors=True,
                                                   trust_remote_code=False,
                                                   device="cuda:0",
                                                   use_triton=False,
                                                   quantize_config=None)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1
        )

        self.local_llm = HuggingFacePipeline(pipeline=pipe)

        retriever = self.vector_db_config.db.as_retriever(search_kwargs=dict(k=1))
        self.memory = VectorStoreRetrieverMemory(retriever=retriever)
