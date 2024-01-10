import os

import torch

from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download

from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GenerationConfig, pipeline

from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME, MODEL_ID, MODEL_BASENAME, DEVICE_TYPE

def load_model(device_type, model_id, model_basename):
    '''
    Select a model on huggingface.

    Running this for the first time, will download the model and cache it for future runs.

    Args:
        device_type: The device type to use for the model. Either "cpu", "cuda" or "mps"
        model_id: Identifier of the model to load from HuggingFace's model hub
        model_basename: Basename of the model if using quantized models
    
    Returns:
        A HuggingFacePipeline instance
    '''
    # Select the Model ID and model_basename (if need be)
    model_id = model_id
    model_basename = model_basename
    
    if model_basename is not None:
        if ".ggml" in model_basename:
            # using Llamacpp for GGML qunatised models
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=model_basename,
                resume_download=True
            )

            max_ctx_size = 2048

            # set the arguments
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size
            }

            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            elif device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            
            return LlamaCpp(**kwargs)

        else:
            # using AutoGPTQForCausalLM for quantized models
            
            if ".safetensors" in model_basename:
                # remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            # using AutoGPTQ for quantised models
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True
            )

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    
    elif device_type.lower() == "cuda":
        # using AutoModelForCausalLM for full models
        tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    
    else:
        # using Llama Tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_id
        )

        model = LlamaForCausalLM.from_pretrained(
            model_id
        )

    # loading configuration from the model to avoid warnings
    # Follow this link for more info: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    generation_config = GenerationConfig.from_pretrained(
        model_id
    )

    # create pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return local_llm


def main(show_resources):
    # load the instruction embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE_TYPE}
    )

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()

    # set up the prompt template
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    {history}

    Question: {question}
    Helpul Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=template
    )
    memory = ConversationBufferMemory(input_key="question", memory_key="history")


    # prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]

    # load the LLM for generating Natural Language responses
    llm = load_model(
        device_type=DEVICE_TYPE,
        model_id=MODEL_ID,
        model_basename=MODEL_BASENAME
    )

    # setup langchain pipeline
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt":prompt, "memory":memory}
    )

    # interactive questions and answers loop
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # get answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # print the query and results
        print("\n\n> Question:")
        print(query)

        print("\n> Answer:")
        print(answer)

        # print the source documents
        if show_resources:
            # this is a flag used to print relevant resources for answers
            print("\n----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("\n----------------------------------SOURCE DOCUMENTS---------------------------")


if __name__ == "__main__":
    main(show_resources=True)