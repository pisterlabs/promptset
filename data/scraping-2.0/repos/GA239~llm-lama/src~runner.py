import os

from langchain.llms import LlamaCpp
from llama_cpp import Llama


def get_local_model_path(model_name) -> str:
    path = "../models/TheBloke/"
    model_path = os.path.join(path, model_name)
    return model_path


def get_cpp_lama(model_name, model_path, lang_chain=False):
    model_depend_args = {"Llama-2-70B-GGML": {"n_gqa": 8}}
    context_size = 3048
    if not lang_chain:
        return Llama(
            model_path=model_path,
            n_threads=7,
            n_ctx=context_size,
            n_gpu_layers=1,  # Metal set to 1 is enough.
            n_batch=512,  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
            **model_depend_args.get(model_name, {"use_mlock": True}),
        )
    return LlamaCpp(
        model_path=model_path,
        temperature=0.0,
        n_threads=7,
        n_ctx=context_size,
        n_gpu_layers=1,  # Metal set to 1 is enough.
        n_batch=512,  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        **model_depend_args.get(model_name, {"use_mlock": True}),
    )


def main():
    m_name = "Llama-2-7b-Chat-GGUF"
    model_file = "llama-2-7b-chat.Q4_K_M.gguf"
    model_path = os.path.join(get_local_model_path(m_name), model_file)
    llm = get_cpp_lama(m_name, model_path, lang_chain=True)
    output = llm("Name the planets in the solar system?", echo=True)
    print(output)


if __name__ == "__main__":
    main()
