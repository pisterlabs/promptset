from typing import Dict,List,Generator,Type,Optional
import openai
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers import AutoModel,AutoTokenizer,AutoModelForCausalLM,GenerationConfig,LlamaTokenizer,LlamaForCausalLM
from llm_rs import Llama,SessionConfig,Precision
from llm_rs import GenerationConfig as RSGenerationConfig
from huggingface_hub import hf_hub_download
import torch
from abc import ABC, abstractmethod
import logging 
from dependency_injector.providers import Configuration  
from schemas.chat import ChatMessage,ModelInfo
from .model_utils import GeneratorStreamer,ManualStopCondition,CPUStreamer
import threading


#gpu only dependencies:
CAN_RUN_PEFT=False
try:
    from peft import PeftModel
    CAN_RUN_PEFT=True
except:
    pass

class ModelAdapter(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def generate(self,messages:List[ChatMessage],generationConfig:GenerationConfig,stop_words:List[str]=[])->str:
        pass
    
    @abstractmethod
    def generate_streaming(self,messages:List[ChatMessage],generationConfig:GenerationConfig,stop_words:List[str]=[])->Generator[str,None,None]:
        pass
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def info(self)->ModelInfo:
        pass
    
    def default_config(self)->GenerationConfig:
        return GenerationConfig()


class ChatGPT_Adapter(ModelAdapter):
    def __init__(self,token:str=None) -> None:
        self.token = token
        self.total_tokens = 0
        self.model_name = "gpt-3.5-turbo"
        if token:
            openai.api_key = token
        else:
            raise Exception("No OpenAI Token Provided! Please provide it over the envirnoment variable OPENAI_TOKEN or use the GPU or CPU models!")
    
    def load(self):
        pass
    
    def info(self)->ModelInfo:
        return ModelInfo(name="OpenAI",model=self.model_name,accelerator="External API")
    
    def default_config(self) -> GenerationConfig:
        return GenerationConfig(temperature=1,top_p=1,repetition_penalty=0,max_new_tokens=256)
    
    def generate(self,messages:List[ChatMessage],generationConfig:GenerationConfig,stop_words:List[str]=[])->str:
        
        transformed_messages = [m.dict() for m in messages]
        result = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=transformed_messages,
                    temperature=generationConfig.temperature,
                    top_p=generationConfig.top_p,
                    frequency_penalty = generationConfig.repetition_penalty,
                    max_tokens=generationConfig.max_new_tokens
                    )
        
        used_tokens = result['usage']['total_tokens']
        self.total_tokens += used_tokens
        logging.info(f"OpenAI: Used {used_tokens} Tokens! Accumulated costs: ({(self.total_tokens/1000)*0.002}$)")
        
        return result['choices'][0]['message']['content']
    
    def generate_streaming(self,messages:List[ChatMessage],generationConfig:GenerationConfig,stop_words:List[str]=[])->Generator[str,None,None]:
        transformed_messages = [m.dict() for m in messages]
        tokens_of_request=0
        for result in openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=transformed_messages,
                    temperature=generationConfig.temperature,
                    top_p=generationConfig.top_p,
                    frequency_penalty = generationConfig.repetition_penalty,
                    max_tokens=generationConfig.max_new_tokens,
                    stream=True
                    ):
            
            delta = result['choices'][0]['delta']
            if "role" in delta:
                continue
            elif "content" in delta:
                yield delta['content']
            else:
                break
            
        self.total_tokens += tokens_of_request
        logging.info(f"OpenAI: Used {tokens_of_request} Tokens! Accumulated costs: ({(self.total_tokens/1000)*0.002}$)")
    
def build_llm_prompt(messages:List[ChatMessage])->str:
    prompt=""
    
    for message in messages:
        clean_content = message.content.strip('\n').strip()
        if message.role == "system":
            prompt += clean_content+"\n"
        elif message.role == "user":
            prompt += f"Human:{clean_content}\n"
        elif message.role == "assistant":
            prompt += f"AI:{clean_content}\n"
        
    if  messages[-1].role == "user":
        prompt += "AI:"
    return prompt
    
class HF_Gpu_Adapter(ModelAdapter):
    def __init__(self,base_model:str,
                 use_peft:bool,
                 adapter_model:str,
                 use_8bit:bool,
                 apply_optimications:bool,
                 max_length:int=2000,
                 model_prototype:Type[AutoModel]=LlamaForCausalLM,
                 tokenizer_prototype:Type[AutoTokenizer]=LlamaTokenizer,
                 tokenizer_name:Optional[str]=None
                 ) -> None:
        self.base_model = base_model
        self.use_peft = use_peft
        self.adapter_model = adapter_model
        self.use_8bit = use_8bit
        self.apply_optimications = apply_optimications
        self.max_length = max_length
        self.model_prototype = model_prototype
        self.tokenizer_prototype = tokenizer_prototype
        self.tokenizer_name = tokenizer_name if (tokenizer_name and len(tokenizer_name) > 0 ) else base_model
        self.stop_reason = None
        
        
        if not torch.cuda.is_available():
            raise Exception("No GPU available! Please use the CPU or OpenAI models!")
     
        
    def info(self)->ModelInfo:
        return ModelInfo(name="Huggingface",model=self.adapter_model if self.use_peft else self.base_model, accelerator="GPU")
       
    def apply_optimizations(self):
        #enable flash attention and tf32 computations
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cudnn.benchmark=True
     
    def default_config(self)->GenerationConfig:
        return GenerationConfig(top_p=0.9,num_beams=1,repetition_penalty=1.1,max_new_tokens=256,use_cache=True)
    
    def load(self):
        if self.apply_optimications:
            self.apply_optimizations()
            
        self.model = self.model_prototype.from_pretrained(self.base_model,
                                                      torch_dtype=torch.float16,
                                                      device_map="auto",
                                                      load_in_8bit=self.use_8bit)
        
        if self.use_peft:
            if not CAN_RUN_PEFT:
                "Peft is not available! Please use the gpu container!"
            self.model = PeftModel.from_pretrained(self.model, self.adapter_model, device_map={'': 0})
            
        self.model = self.model.eval()
        
        if self.apply_optimications:
            self.model = torch.compile(self.model,mode="max-autotune")
            
        self.tokenizer = self.tokenizer_prototype.from_pretrained(self.tokenizer_name)
        self.tokenizer.max_length = self.max_length
            
            
    def generate(self,messages:List[Dict[str,str]],generationConfig:GenerationConfig,stop_words:List[str]=[])->str:
        prompt=build_llm_prompt(messages)
        #These are only used here for the stopword detection
        manual_stop = ManualStopCondition()
        streamer = GeneratorStreamer(self.tokenizer,manual_stop,stop_words=stop_words)
        
        with torch.no_grad():
            input = self.tokenizer(prompt, return_tensors="pt")
            input_ids = input["input_ids"].to("cuda")
            generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generationConfig,
                    return_dict_in_generate=True,
                    output_scores=False,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([manual_stop])    
                )
            generated_tokens = generation_output.sequences[0]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
        if manual_stop.should_stop.is_set():
            self.stop_reason="Stopword detected!"
        else:
            self.stop_reason="Max Tokens!"
            
        return generated_text[len(prompt):]
    
    def generate_streaming(self,messages:List[ChatMessage],generationConfig:GenerationConfig,stop_words:List[str]=[])->Generator[str,None,None]:
        
        prompt=build_llm_prompt(messages)
        manual_stop = ManualStopCondition()
        streamer = GeneratorStreamer(self.tokenizer,manual_stop,stop_words=stop_words)
        
        with torch.no_grad():
            input = self.tokenizer(prompt, return_tensors="pt")
            input_ids = input["input_ids"].to("cuda")
            thread = threading.Thread(target=self.model.generate,
                    kwargs={
                        "input_ids":input_ids,
                        "generation_config":generationConfig,
                        "streamer":streamer,
                        "stopping_criteria":StoppingCriteriaList([manual_stop])              
                    },
                    daemon=True)
            thread.start()
            
        yield from streamer
            
        if manual_stop.should_stop.is_set():
            self.stop_reason="Stopword detected!"
        else:
            self.stop_reason="Max Tokens!"
    
class Cpu_Adapter(ModelAdapter): 
    def __init__(self,hf_token:str=None,repository:str="Sosaka/Alpaca-native-4bit-ggml",filename:str="ggml-alpaca-7b-q4.bin",max_length:int=2048,threads:int=8,kv_16:bool=True,mmap:bool=True) -> None:
        self.max_length = max_length
        self.threads=threads
        self.hf_token=hf_token
        self.repository=repository
        self.filename = filename
        self.kv_16=kv_16
        self.mmap=mmap
           
    def info(self)->ModelInfo:
        return ModelInfo(name="llm-rs",model=self.repository, accelerator="CPU")
         
    def default_config(self)->GenerationConfig:
        return GenerationConfig(top_p=0.9,top_k=40,temperature=0.8,repetition_penalty=1.1,max_new_tokens=256)
    
    def load(self):
        self.ggjt_model = hf_hub_download(repo_id=self.repository, filename=self.filename,token=self.hf_token)  
        precision = Precision.FP16 if self.kv_16 else Precision.FP32
        self.session_config = SessionConfig(threads=self.threads,context_length=self.max_length,keys_memory_type=precision,values_memory_type=precision,prefer_mmap=self.mmap)   
        self.model = Llama(str(self.ggjt_model),session_config=self.session_config,verbose=True)
    
    
    def _hf_to_rs_config(self,generationConfig:GenerationConfig)->RSGenerationConfig:
        return RSGenerationConfig(
            top_k=generationConfig.top_k,
            top_p=generationConfig.top_p,
            temperature=generationConfig.temperature,
            repetition_penalty=generationConfig.repetition_penalty,
            max_new_tokens=generationConfig.max_new_tokens,
        )
        
        
    def generate(self,messages:List[ChatMessage],generationConfig:GenerationConfig,stop_words:List[str]=[])->str:
        words = list(self.generate_streaming(messages,generationConfig,stop_words=stop_words))
        return "".join(words)
    
    def generate_streaming(self,messages:List[ChatMessage],generationConfig:GenerationConfig,stop_words:List[str]=[])->Generator[str,None,None]:
        prompt=build_llm_prompt(messages)
        config = self._hf_to_rs_config(generationConfig)
        
        streamer = CPUStreamer(self.model,config=config,prompt=prompt,stop_words=stop_words)
        streamer.start()
        yield from streamer
        

        
def adapter_factory(configuration:Configuration)->ModelAdapter:
    model_to_use = configuration["chatmodel"]
    if model_to_use == "OPENAI":
        return ChatGPT_Adapter(configuration["open_ai_token"])
    elif model_to_use == "GPU":
        return HF_Gpu_Adapter(
            base_model=configuration["base_chat_model"],
            use_peft=configuration["use_peft"],
            adapter_model=configuration["adapter_chat_model"],
            use_8bit=configuration["use_8bit"],
            apply_optimications=configuration["chat_apply_optimizations"],
            max_length=configuration["chat_max_length"]
        )
    elif model_to_use == "CPU":
        return Cpu_Adapter(
            hf_token=configuration["hf_token"],
            threads=configuration["cpu_model_threads"],
            repository=configuration["cpu_model_repo"],
            filename=configuration["cpu_model_filename"],
            max_length=configuration["chat_max_length"],
            kv_16=configuration["cpu_model_kv_16"],
            mmap=configuration["cpu_model_mmap"]
            )
    else:
        raise Exception("Unknown model type: " + model_to_use)
        
            