import transformers
from torch import bfloat16, cuda
from langchain.llms import HuggingFacePipeline
import os



class MyHuggingFaceModel():

    def __init__(self,hf_id):
        self.model = hf_id

    def _text_pipeline(self, llm_model_id):
        bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                                    bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=bfloat16)
        hf_auth = os.getenv('HF_AUTH_TOKEN')
        model_config = transformers.AutoConfig.from_pretrained(llm_model_id,use_auth_token = hf_auth)
        llm_model =  transformers.AutoModelForCausalLM.from_pretrained(llm_model_id,trust_remote_code=True,
                                                                        config=model_config,quantization_config=bnb_config,
                                                                        device_map='auto',use_auth_token=hf_auth)
        tokenizer = transformers.AutoTokenizer.from_pretrained(llm_model_id,use_auth_token=hf_auth)
        generate_text = transformers.pipeline(model=llm_model, tokenizer=tokenizer,return_full_text=True, 
                                        task='text-generation',temperature=0.1, max_new_tokens=512, repetition_penalty=1.1)
        return generate_text
    
    def create_langchain_model(self):
        text_pipeline = self._text_pipeline(self.model)
        llm = HuggingFacePipeline(pipeline=text_pipeline)
        return llm