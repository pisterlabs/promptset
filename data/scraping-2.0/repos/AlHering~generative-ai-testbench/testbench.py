# -*- coding: utf-8 -*-
"""
****************************************************
*             generative_ai_testbench                
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from src.configuration import configuration as cfg
from pydantic.error_wrappers import ValidationError
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Setup prompting
promt_template = """
Question: {question}
Answer: Please devide your answer in multiple steps and explain each step to be sure we have the right answer.
"""
prompt = PromptTemplate(template=promt_template, input_variables=["question"])

# Instance of callback manager for token-wise streaming
callback_manager = CallbackManager(
    [StreamingStdOutCallbackHandler()])

# os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH, "vicuna-7B-v1.3-GGML/vicuna-7b-v1.3.ggmlv3.q5_0.bin")
# working with -> lama-cpp-python==0.1.64 langchain==0.0.197 pygpt4all==1.1.0 pydantic==1.10.9 typing-inspect==0.9.0 typing_extensions==4.7.0
# os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH, "vicuna-7B-v1.3-GGML/vicuna-7b-v1.3.ggmlv3.q5_0.bin")
# working with -> lama-cpp-python==0.1.64 langchain==0.0.197 pygpt4all==1.1.0 pydantic==1.10.9 typing-inspect==0.9.0 typing_extensions==4.7.0
# "/media/linux/Data II/text_generation_models/eachadea_ggml-vicuna-7b-1.1/ggml-vicuna-7b-1.1-q4_0.bin"
# working with -> llama-cpp-python==0.1.48 pydantic==1.9.0 typing-inspect==0.8.0 typing-extensions>=4.5.0 langchain==0.0.197
# working with -> lama-cpp-python==0.1.48 langchain==0.0.173 pygpt4all==1.1.0
# "/media/linux/Data II/text_generation_models/eachadea_ggml-vicuna-7b-1.1/ggml-vicuna-7b-1.1-q4_1.bin"
# working with -> llama-cpp-python==0.1.48 pydantic==1.9.0 typing-inspect==0.8.0 typing-extensions=4.5.0 pygpt4all==1.1.0 langchain==0.0.197
# working with -> lama-cpp-python==0.1.48 langchain==0.0.173 pygpt4all==1.1.0
# os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH, "orca_mini_7B-GGML/orca-mini-7b.ggmlv3.q4_1.bin")
#

# Setup central LLM
llm = LlamaCpp(
    model_path=os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                            "orca_mini_7B-GGML/orca-mini-7b.ggmlv3.q4_1.bin"),
    callback_manager=callback_manager,
    verbose=True)

# Setup basic chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Run Chain
question = "What topics are important for Data Engineering and why?"
llm_chain.run(question)

"""
->
llama.cpp: loading model from *****************/eachadea_ggml-vicuna-7b-1.1/ggml-vicuna-7b-1.1-q4_0.bin
llama_model_load_internal: format     = ggjt v1 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: ftype      = 4 (mostly Q4_1, some F16)
llama_model_load_internal: n_ff       = 11008
llama_model_load_internal: n_parts    = 1
llama_model_load_internal: model size = 7B
llama_model_load_internal: ggml ctx size =  68.20 KB
llama_model_load_internal: mem required  = 5809.33 MB (+ 1026.00 MB per state)
llama_init_from_file: kv self size  =  256.00 MB
AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | 

Step 1: What is Data Engineering?
Data engineering is a domain of data science that deals with designing, building, testing, and deploying systems for managing large-scale data processing, data storage, and data delivery. It involves the following aspects:
* Designing the architecture of the system, including data pipelines, databases, and data warehouses.
* Building data models to handle different types of data, such as structured, semi-structured, and unstructured data.
* Testing and validating the performance and scalability of the system.
* Deploying the system in a production environment, including monitoring its health and performance.
Step 2: Why are these topics important for Data Engineering?
These topics are important for Data Engineering because they help to build efficient and reliable systems that can handle large amounts of data with high throughput and low latency. By addressing various aspects of the system, such as data pipeline design, data modeling, testing, and deployment, data engineers can ensure the reliability and scalability of their systems, which is crucial for handling big data.
"""
