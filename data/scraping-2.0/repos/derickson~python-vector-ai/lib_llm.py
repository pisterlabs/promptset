## for conversation LLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, StoppingCriteria, StoppingCriteriaList
import gc
import os

from lib_webLLM import WebLLM

OPTION_CUDA_USE_GPU = os.getenv('OPTION_CUDA_USE_GPU', 'False') == "True"
cache_dir = "../cache"

def clean_cache(p=False):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if p:
        print(f"TORCH CUDA MEMORY ALLOCATED: {torch.cuda.memory_allocated()/(1024)} Kb")

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def getStableLM3B():
    model_id = 'stabilityai/stablelm-tuned-alpha-3b'
    print(f">> Prep. Get {model_id} ready to go")
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    clean_cache()
    model = AutoModelForCausalLM.from_pretrained(model_id)
    clean_cache()
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=256,
        temperature=0.7,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        pad_token_id=50256, num_return_sequences=1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm



def getFlanLarge():
    
    model_id = 'google/flan-t5-large'
    print(f">> Prep. Get {model_id} ready to go")
    # model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    if OPTION_CUDA_USE_GPU:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir, load_in_8bit=True, device_map='auto') 
            model.cuda()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir) 
    
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=100
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

## options are flan and stablelm
MODEL = "flan"


if MODEL == "flan":
    local_llm = getFlanLarge()
elif MODEL == "webflan":
    local_llm = WebLLM()
else:
    local_llm = getStableLM3B()


def make_the_llm():
    if MODEL == "flan"  or MODEL == "webflan":
        template_informed = """
        I am a helpful AI that answers questions. When I don't know the answer I say I don't know. 
        I know context: {context}
        when asked: {question}
        my response using only information in the context is: """
    else:
        stablelm_system_prompt = """
            - StableLM answers questions using only the information {context}
            - If it does not have the information, StableLM answers with 'I do not know'
        """
        template_informed  = f"<|SYSTEM|>{stablelm_system_prompt}"+"<|USER|>{question}?<|ASSISTANT|>"

    prompt_informed = PromptTemplate(template=template_informed, input_variables=["context", "question"])

    return LLMChain(prompt=prompt_informed, llm=local_llm)


def make_the_llm_ignorant():
    if MODEL == "flan" or MODEL == "webflan":
        template_ignorant = "{question}"
    else:
        stablelm_system_prompt = """
            - StableLM answers questions 
            - If it does not have the information, StableLM answers with 'I do not know'
        """
        template_ignorant  = f"<|SYSTEM|>{stablelm_system_prompt}"+"<|USER|>{question}?<|ASSISTANT|>"

    prompt_ignorant = PromptTemplate(template=template_ignorant, input_variables=["question"])

    return LLMChain(prompt=prompt_ignorant, llm=local_llm)