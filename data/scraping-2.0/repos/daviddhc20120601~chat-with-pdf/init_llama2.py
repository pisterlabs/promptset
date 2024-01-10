# For download the models
# !pip install huggingface_hub

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format


from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)


# !pip install llama-cpp-python
# CPU
from llama_cpp import Llama

lcpp_llm = Llama(
    model_path=model_path,
    n_threads=64, # CPU cores
    )

prompt = "Write a linear regression in python"
prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

USER: {prompt}

ASSISTANT:
'''

response = lcpp_llm(
    prompt=prompt_template,
    max_tokens=256,
    temperature=0.5,
    top_p=0.95,
    repeat_penalty=1.2,
    top_k=50,
    echo=True
    )

print(response["choices"][0]["text"])

# Inference with langchain

# !pip -q install langchain

lcpp_llm.reset()
lcpp_llm.set_cache(None)
lcpp_llm = None
del lcpp_llm

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download

template = """USER: {question}
ASSISTANT: Let's work this out in a step by step way to be sure we have the right answer."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Loading model,
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=1024,
    # n_gpu_layers=n_gpu_layers,
    # n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Write a linear regression in python"

llm_chain.run(question)