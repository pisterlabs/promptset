#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
from torch import cuda as torch_cuda
import torch

import psutil

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = 2000
is_gpu_enabled = (os.environ.get('IS_GPU_ENABLED', 'False').lower() == 'true')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
from constants import CHROMA_SETTINGS

def get_gpu_memory() -> int:
    """
    Returns the amount of free memory in MB for each GPU.
    """
    return int(torch_cuda.mem_get_info()[0]/(1024**2))

def calculate_layer_count() -> int | None:
    """
    Calculates the number of layers that can be used on the GPU.
    """
    if not is_gpu_enabled:
        return None
    LAYER_SIZE_MB = 120.6 # This is the size of a single layer on VRAM, and is an approximation.
    # The current set value is for 7B models. For other models, this value should be changed.
    LAYERS_TO_REDUCE = 6 # About 700 MB is needed for the LLM to run, so we reduce the layer count by 6 to be safe.
    if (get_gpu_memory()//LAYER_SIZE_MB) - LAYERS_TO_REDUCE > 32:
        return 32
    else:
        return (get_gpu_memory()//LAYER_SIZE_MB-LAYERS_TO_REDUCE)

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings_kwargs = {'device': 'cuda'} if is_gpu_enabled else {}
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=embeddings_kwargs)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path='/data/privateGPT-gpu/models/llama-2-7b-chat.ggmlv3.q4_0.bin', n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, n_gpu_layers=calculate_layer_count())
            #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
        case "GPT4All":
            if is_gpu_enabled:
                print("GPU is enabled, but GPT4All does not support GPU acceleration. Please use LlamaCpp instead.")
                exit(1)
            llm = GPT4All(model='/data/privateGPT-gpu/models/ggml-gpt4all-j-v1.3-groovy.bin', n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
            #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
        case "MedLlama":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPT-gpu/models/medllama', task="text-generation", device=1,
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
        case "phi":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPT-gpu/models/phi-1_5',task="text-generation", 
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
        case "codegeex2":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPT-gpu/models/codegeex2-6b', task="text-generation", device=1,
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
        case "codellama":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPT-gpu/models/CodeLlama-7b-hf', task="text-generation", device=1,
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
        case "vicuna":
            llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPT-gpu/models/vicuna-7b-v1.5', task="text-generation", device=1,
                                        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx}) 
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        if model_type == "codegeex2":
            query = "# language: Python\n# "
            query += input("Enter a query: ")
            query += "\n"
            if query == "exit":
                break
        else:
            query = input("\nEnter a query: ")
            if query == "exit":
                break

        '''# Get the answer from the chain
        if model_type == "phi" or model_type == "MedLlama":
            # Tokenize the input text
            batch = tokenizer(
                query,
                return_tensors="pt", 
                add_special_tokens=False,
            ).to(llm.device)
            with torch.no_grad():
                generated = llm.generate(inputs = batch["input_ids"], max_length=256, do_sample=True, top_k=1)
                answer = tokenizer.decode(generated[0])
        else:
            res = qa(query)
            answer, docs = res['result'], [] if args.hide_source else res['source_documents']'''
        
        import time
        start = time.time()
        res = qa(query)
        end = time.time()
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        
        # Print the vram the model is using
        print(f"VRAM usage: {torch.cuda.memory_allocated()/1024**2} MB")
        # Create a log file where the VRAM usage and the corresponding and model type and  are stored
        with open("/data/privateGPT-gpu/logs/vram_usage.log", "a") as f:
            f.write(f"{torch.cuda.memory_allocated()//1024**2} MB\t{psutil.virtual_memory().used//1024**2} MB\t{psutil.cpu_percent()}%\t{model_type}\t{model_n_ctx}\t{int(end - start)}\n")

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        if model_type == "codegeex2":
            '''from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("/data/privateGPT-gpu/models/codegeex2-6b", trust_remote_code=True)
            answer = tokenizer.decode(answer)'''
            print(answer)
        else:
            print(answer)

        '''# Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)'''

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
