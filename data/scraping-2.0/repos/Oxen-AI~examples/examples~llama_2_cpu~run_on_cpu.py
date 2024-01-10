from langchain.llms import CTransformers
from langchain.callbacks.base import BaseCallbackHandler

# Handler that prints each new token as it is computed
class NewTokenHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True)

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(
    model="models/llama-2-7b-chat.ggmlv3.q8_0.bin", # Location of downloaded GGML model
    model_type="llama", # Model type Llama
    stream=True,
    callbacks=[NewTokenHandler()],
    config={'max_new_tokens': 256, 'temperature': 0.01}
)

# Accept user input
while True:
    prompt = input('> ')
    output = llm(prompt)
    print(output)