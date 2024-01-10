from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
import datetime

#llm = OpenAI(temperature=0)
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 128  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 80  # Metal set to 1 is enough.
n_batch = 128  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!

llm = LlamaCpp(
    model_path="/Workspace/yamada/pretrain/Llama-2-13B-chat-GGML/llama-2-13b-chat.ggmlv3.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=500,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

start = datetime.datetime.now()
print(f'start:{start}')
from tqdm import tqdm
for i in tqdm(range(10)):
    llm.predict("Who are you")
    end = datetime.datetime.now()
    print(f'end:{end}')
print(f'duration:{(end-start).total_seconds()} seconds, start:{start}, end:{end}')

"""
CUDA_VISIBLE_DEVICES=0 python test/LlamaCppTest.py

"""
