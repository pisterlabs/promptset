# 0: Import ray serve and request from starlette
from ray import serve
from starlette.requests import Request
from langchain.chains import LLMChain
from langchain.llms import Ollama, LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.


# 1: Define a Ray Serve deployment.
@serve.deployment
class LLMServe:
	def __init__(self) -> None:
		# All the initialization code goes here
		pass
		llamacpp = LlamaCpp(
		# model_path="/home/six/llms/gguf/openhermes-2-mistral-7b.Q4_K_M.gguf",
		# model_path="/llms/gguf/openhermes-2.5-neural-chat-v3-3-slerp.Q6_K.gguf",
		model_path="/llms/gguf/dolphin-2.6-mistral-7b.Q5_K_M.gguf",
		n_gpu_layers=n_gpu_layers,
		n_batch=n_batch,
		n_ctx=4096,
		f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
		# callback_manager=callback_manager,
		verbose=False,  # Verbose is required to pass to the callback manager
		temperature=0,
	)

		prompt = PromptTemplate.from_template("""<|im_start|>system
		You are an AI assistant, who can be referred to as Jarvis or Józsi. You can speak English and Hungarian. Your task is to analyze messages to determine if you are being addressed directly. You should look for your name (Jarvis or Józsi).
		<|im_start|>user
		{prompt}<|im_end|>
		<|im_start|>assistant
		""")
		self.llm = {"prompt": RunnablePassthrough()} | prompt | llamacpp
		# self.chain = LLMChain(llm=llamacpp, prompt=prompt)

	def _run_chain(self, text: str):
		print('text:', text)
		return self.llm.invoke(text)
	
	async def __call__(self, request: Request):
		# 1. Parse the request
		text = request.query_params["text"]
		# 2. Run the chain
		resp = self._run_chain(text)
		# 3. Return the response
		return resp["text"]

# 2: Bind the model to deployment
deployment = LLMServe.bind()

# 3: Run the deployment
serve.api.run(deployment, port=8282)
#%%
import requests

text = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
response = requests.post(f"http://localhost:8282/?text={text}")
print(response.content.decode())