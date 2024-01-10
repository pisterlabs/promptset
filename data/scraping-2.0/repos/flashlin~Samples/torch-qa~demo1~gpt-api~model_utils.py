from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
import torch
import functools
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_lit import StreamDisplayHandler


def create_llama2():
    model_name = "CodeLlama-7B-Instruct-GGUF/codellama-7b-instruct.Q4_K_M.gguf"
    model_name = "Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return LlamaCpp(
        model_path=f"../../models/{model_name}",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        n_ctx=2048,  # 請求上下文 ValueError: Requested tokens (1130) exceed context window of 512
        callback_manager=callback_manager,
        verbose=False,  # True
        streaming=True,
    )


def create_llama2_v2(callbackHandler):
    model_name = "CodeLlama-7B-Instruct-GGUF/codellama-7b-instruct.Q4_K_M.gguf"
    model_name = "Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
    model_name = "TheBloke_Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler(), StreamDisplayHandler(callbackHandler)])
    return LlamaCpp(
        model_path=f"./models/{model_name}",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        n_ctx=2048,  # 請求上下文 ValueError: Requested tokens (1130) exceed context window of 512
        callback_manager=callback_manager,
        verbose=False,  # True
        streaming=True,
    )


def time_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        return (result, exec_time)
    return wrapper


def memory_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result, exec_time = func(*args, **kwargs)
        peak_mem = torch.cuda.max_memory_allocated()
        peak_mem_consumption = peak_mem / 1e9
        return peak_mem_consumption, exec_time, result
    return wrapper


@memory_decorator
@time_decorator
def generate_output1(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")
    outputs = model.generate(input_ids, max_length=500)
    output_text = tokenizer.decode(outputs[0])
    return output_text


@memory_decorator
@time_decorator
def generate_output2(prompt: str, llm) -> str:
    output_text = llm.generate(prompt)
    return output_text