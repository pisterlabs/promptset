#%%
from functools import lru_cache
import os
from langchain.llms import LlamaCpp, Ollama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


@lru_cache(maxsize=None) 
def get_llm_model_memoized(model_path):
	n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
	n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
	return LlamaCpp(
		model_path=model_path,
		n_gpu_layers=n_gpu_layers,
		n_batch=n_batch,
		n_ctx=4096,
		f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
		# callback_manager=callback_manager,
		verbose=False,  # Verbose is required to pass to the callback manager
		temperature=0,
	)

def get_ollama_llm_model(model_name="openhermes2.5-mistral"):
	ollama_url = os.environ.get('OLLAMA_URL')
	llm = Ollama(
		base_url=ollama_url, model=model_name, temperature=0.0
		# base_url="https://llm.sixzero.xyz", model="codellama:13b", temperature=0.0
		# base_url="https://llm.sixzero.xyz", model="starling-lm", temperature=0.1
	)
	return llm

def get_llm_model(model_path="/llms/gguf/dolphin-2.6-mistral-7b.Q5_K_M.gguf", system="You can be referred to as Jarvis or Józsi. You can speak English and Hungarian. Your task is to analyze messages to determine if you are being addressed directly. You should look for your name (Jarvis or Józsi). Respond with 'Yes' if you are directly addressed in any part of the message, and 'No' if you are not."):

		# model_path="/home/six/llms/gguf/openhermes-2-mistral-7b.Q4_K_M.gguf",
		# model_path="/llms/gguf/openhermes-2.5-neural-chat-v3-3-slerp.Q6_K.gguf",
	# talk system initialization flag
	llamacpp = get_llm_model_memoized(model_path)

	# f"""Message: {text}\n 
	# Analyze the message to determine if it contains a direct reference to you (as Jarvis or Józsi) in any sentence. "Jarvis." is a good reference.  """ +
	# 					  # " Please reason why you think you were mentioned. " +
	# "\n"
	prompt_ = PromptTemplate.from_template(f"""<|im_start|>system
	You are an AI assistant. {system}
	<|im_start|>user\n"""
	"""{prompt}<|im_end|>
	<|im_start|>assistant
	""")
	prompt_ = PromptTemplate.from_template(f"""<|im_start|>system
	You are an AI assistant. {system}
	<|im_start|>user\n""" +
	"""{prompt}<|im_end|>
	<|im_start|>assistant
	""")
	llm = {"prompt": RunnablePassthrough()} | prompt_ | llamacpp

	return llm