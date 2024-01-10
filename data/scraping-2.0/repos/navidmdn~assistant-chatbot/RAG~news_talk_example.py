import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import fire
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.document_loaders import AsyncChromiumLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from operator import itemgetter
from transformers import BitsAndBytesConfig


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}\nall model parameters:"
          f" {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")


def test_model(tokenizer, model):
    messages = [
        {"role": "user", "content": "What is your favourite movie?"},
        {"role": "assistant",
         "content": "Well, I'm quite partial to a good documentary. What about you?"},
        {"role": "user", "content": "I like action movies. what about you?"},
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    # model.to(device)
    outputs = model.generate(inputs, max_new_tokens=1000, do_sample=True)
    print(tokenizer.decode(outputs[0]))


def run(model_name='gpt2', load_in_4bit=False, cache_dir=None, vector_store_path='vector_ds/test_wiki.faiss',
        vector_store_embedding_model='sentence-transformers/all-mpnet-base-v2'):

    #################################################################
    # Tokenizer
    #################################################################

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #################################################################
    # bitsandbytes parameters
    #################################################################
    if load_in_4bit:
        # Activate 4-bit precision base model loading
        use_4bit = True

        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"

        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"

        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False
        #################################################################
        # Set up quantization config
        #################################################################
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
    else:
        bnb_config = None


    #################################################################
    # Load pre-trained config
    #################################################################
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
    )

    print_number_of_trainable_model_parameters(llm)
    # test_model(tokenizer, model)

    pipe = pipeline('text-generation', model=llm, tokenizer=tokenizer, max_length=1000)
    local_llm = HuggingFacePipeline(pipeline=pipe)

    embedding_model = HuggingFaceEmbeddings(model_name=vector_store_embedding_model)
    db = FAISS.load_local(vector_store_path, embedding_model)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Create prompt template
    prompt_template = """
    ### [INST] Instruction: Answer the question based on your general knowledge. Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
     """

    contex_combiner = lambda x: "\n".join([y.page_content for y in x])

    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=local_llm, prompt=prompt, verbose=True)
    chain = (
        {"context": retriever | contex_combiner, "question": RunnablePassthrough()}
        | llm_chain
        # | StrOutputParser()
    )

    inp = ""
    while inp != 'exit':
        inp = input(">> ")
        print(chain.invoke(inp))


if __name__ == '__main__':
    # model_name='mistralai/Mistral-7B-Instruct-v0.1'
    fire.Fire(run)
