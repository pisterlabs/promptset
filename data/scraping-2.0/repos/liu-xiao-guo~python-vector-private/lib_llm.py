## for conversation LLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM


def make_the_llm():
    # Get Offline flan-t5-large ready to go, in CPU mode
    print(">> Prep. Get Offline flan-t5-large ready to go, in CPU mode")
    model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id) #load_in_8bit=True, device_map='auto'
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=100
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    # template_informed = """
    # I know the following: {context}
    # Question: {question}
    # Answer: """

    template_informed = """
    I know: {context}
    when asked: {question}
    my response is: """

    prompt_informed = PromptTemplate(template=template_informed, input_variables=["context", "question"])

    return LLMChain(prompt=prompt_informed, llm=local_llm)