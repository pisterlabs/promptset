from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

model_id = "TheBloke/Llama-2-7B-chat-GGML"
model_file="llama-2-7b-chat.ggmlv3.q5_K_M.bin"
config = {'context_length':2048,'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0.1, 'stream': True}

def build_llm(stream_callback):
    return CTransformers(model=model_id,
                    model_file=model_file,
                    model_type="llama",
                    lib='avx2',
                    config=config,
                    callbacks=[stream_callback])