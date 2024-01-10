import logging

from ChromaProcess import load_local_chroma_db
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from HVConstants import (
    DEVICE_TYPE,
    MODEL_ID,
    MODEL_BASENAME
)


# load the LLM for generating Natural Language responses
def load_model(device_type, model_id, model_basename=None):
    """
    from localGPT
    """

    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        # The code supports all huggingface models that ends with GPTQ
        # and have some variation of .no-act.order or .safetensors in their HF repo.
        print("Using AutoGPTQForCausalLM for quantized models")

        if ".safetensors" in model_basename:
            # Remove the ".safetensors" ending if present
            model_basename = model_basename.replace(".safetensors", "")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        logging.info("Tokenizer loaded")

        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            # model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=True,
            inject_fused_mlp=False,
            quantize_config=None,
        )
    elif (
            device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin file in their HF repo.
        print("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        model.tie_weights()
    else:
        print("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
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

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


if DEVICE_TYPE == "cpu":
    llm = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID)
else:
    llm = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)


def answer_query_from_db(query, custom_format=None, search_kwargds=None):
    db = load_local_chroma_db()
    if not db:
        print("No existing chroma db!")
    if search_kwargds is None:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    else:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                         retriever=db.as_retriever(search_kwargs=search_kwargds))  # filter by document
    qa.return_source_documents = True
    res = qa(query)
    answer, docs = res["result"], res["source_documents"]
    if custom_format:
        answer = refine_answer_with_custom_format(answer, custom_format, llm)
    return {
        "answer": answer,
        "source_documents": docs,
        "query": query,
        "custom_format": custom_format
    }


def refine_answer_with_custom_format(answer, custom_format, llm):
    template_string = '''
    Please convert the following target text to the required format below.
    
    target text:{answer}
    
    required format:{custom_format}
    
    '''
    prompt = PromptTemplate.from_template(template_string)
    input_ = prompt.format(answer=answer, custom_format=custom_format)
    output_ = llm(input_)

    return output_


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    result = answer_query_from_db("When is the constitution of freedom of speech created?",
                                  custom_format="please present the answer in the format of YYYY:MM:DD")
    print(result)
