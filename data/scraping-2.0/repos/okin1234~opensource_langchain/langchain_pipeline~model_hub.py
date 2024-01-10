import os
import types
from class_resolver import ClassResolver

from langchain import OpenAI
from .huggingface_model import HuggingFacePipeline
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

MODEL_KWARGS = {"temperature": 0, "max_new_tokens": 2048}
LOAD_KWARGS = {"use_safetensors": True, "trust_remote_code": True, "quantize_config": None, "use_triton": True}

def load_model_gptq(model_id, model_basename, load_kwargs=None):
    model_kwargs = MODEL_KWARGS
    llm = HuggingFacePipeline.from_model_id(model_id, task="text-generation", model_kwargs=model_kwargs, use_gptq=True, model_basename=model_basename, load_kwargs=load_kwargs)
    return llm

def llama_v2_7b_gptq(device="all"):
    model_id = "TheBloke/Llama-2-7B-GPTQ"
    model_basename = "gptq_model-4bit-128g"
    load_kwargs = LOAD_KWARGS
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs)
    return llm

def llama_v2_13b_gptq(device="all"):
    model_id = "TheBloke/Llama-2-13B-GPTQ"
    model_basename = "gptq_model-4bit-128g"
    load_kwargs = LOAD_KWARGS
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def llama_v2_13b_chat_gptq(device="all"):
    model_id = "TheBloke/Llama-2-13B-chat-GPTQ"
    model_basename = "gptq_model-4bit-128g"
    load_kwargs = LOAD_KWARGS
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def orca_mini_v2_13b_gptq(device="all"):
    model_id = "TheBloke/orca_mini_v2_13b-GPTQ"
    model_basename = "orca_mini_v2_13b-GPTQ-4bit-128g.no-act.order"
    load_kwargs = LOAD_KWARGS
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def falcon_7b_instruct_gptq(device="all"):
    # 똥임
    model_id = "TheBloke/falcon-7b-instruct-GPTQ"
    model_basename = "gptq_model-4bit-64g"
    load_kwargs = LOAD_KWARGS
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def wizard_vicuna_7b_uncensored_superhot_gptq(device="all"):
    model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-SuperHOT-8K-GPTQ"
    model_basename = "wizard-vicuna-7b-uncensored-superhot-8k-GPTQ-4bit-128g.no-act.order"
    load_kwargs = LOAD_KWARGS
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def wizard_vicuna_13b_uncensored_superhot_gptq(device="all"):
    model_id = "TheBloke/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GPTQ"
    model_basename = "wizard-vicuna-13b-uncensored-superhot-8k-GPTQ-4bit-128g.no-act.order"
    load_kwargs = LOAD_KWARGS
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def wizard_vicuna_30b_uncensored_gptq(device="all"):
    model_id = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
    model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit--1g.act.order"
    load_kwargs = LOAD_KWARGS
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def llama_v2_13b_guanaco_qlora_gptq(device="all"):
    model_id = "TheBloke/llama-2-13B-Guanaco-QLoRA-GPTQ"
    model_basename = "gptq_model-4bit-128g"
    load_kwargs = LOAD_KWARGS
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def openassistant_llama_v2_13b_orca_8k_gptq(device="all"):
    model_id = "TheBloke/OpenAssistant-Llama2-13B-Orca-8K-3319-GPTQ"
    model_basename = "gptq_model-4bit-128g"
    load_kwargs = {"use_safetensors": True, "trust_remote_code": False, "quantize_config": None, "use_triton": True}
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def stablebeluga_13b_gptq(device="all"):
    model_id = "TheBloke/StableBeluga-13B-GPTQ"
    model_basename = "gptq_model-4bit-128g"
    load_kwargs = {"use_safetensors": True, "trust_remote_code": False, "quantize_config": None, "use_triton": True}
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def stablebeluga_v2_70b_gptq(device="all"):
    model_id = "TheBloke/StableBeluga2-70B-GPTQ"
    model_basename = "gptq_model-4bit--1g"
    load_kwargs = {"inject_fused_attention": False, "use_safetensors": True, "trust_remote_code": False, "quantize_config": None, "use_triton": True}
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def llama_v2_70b_chat_gptq(device="all"):
    model_id = "TheBloke/Llama-2-70B-chat-GPTQ"
    model_basename = "gptq_model-4bit--1g"
    load_kwargs = {"inject_fused_attention": False, "use_safetensors": True, "trust_remote_code": False, "quantize_config": None, "use_triton": True}
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def llama_v2_70b_gptq(device="all"):
    model_id = "TheBloke/Llama-2-70B-GPTQ"
    model_basename = "gptq_model-4bit--1g"
    load_kwargs = {"inject_fused_attention": False, "use_safetensors": True, "trust_remote_code": False, "quantize_config": None, "use_triton": True}
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def llama_v2_70b_instruct_v2_gptq(device="all"):
    model_id = "TheBloke/Upstage-Llama-2-70B-instruct-v2-GPTQ"
    model_basename = "gptq_model-4bit--1g"
    load_kwargs = {"use_safetensors": True, "trust_remote_code": False, "quantize_config": None, "use_triton": True}
    load_kwargs = device_setting(device, load_kwargs=load_kwargs)
    llm = load_model_gptq(model_id, model_basename, load_kwargs=load_kwargs)
    return llm

def gpt():
    llm = OpenAI(temperature=0)
    return llm


def llm_resolver(model_name, device="all"):
    class Base: pass

    func_list = []
    for name, val in globals().items():
        if isinstance(val, types.FunctionType):
            func_list.append(val)
            
    resolver = ClassResolver(func_list, base=Base)
    model = resolver.make(model_name, device=device)
    return model


def device_setting(device, load_kwargs=None):
    if device == "all":
        if load_kwargs:
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs = {"device_map": "auto"}
    else:
        if load_kwargs:
            load_kwargs["device"] = f"cuda:{device}"
        else:
            load_kwargs = {"device": f"cuda:{device}"}
            
    return load_kwargs