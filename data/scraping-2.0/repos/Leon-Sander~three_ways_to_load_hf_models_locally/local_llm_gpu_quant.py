from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, MistralForCausalLM
from langchain.prompts import PromptTemplate

def create_prompt():
    template = """You are a question answering Large Language Model.
    Answer the following question.
    USER: {question}
    ASSISTANT:"""
    prompt = PromptTemplate.from_template(template)
    return prompt

def create_llm_pipeline(model_id,device_map="cuda:0", load_in_8bit=False, load_in_4bit=False):                                     
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map , load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit)
    #model = MistralForCausalLM.from_pretrained(model_id, device_map=device_map , load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit)
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
    print("HF Pipeline created")
    llm_pipeline = HuggingFacePipeline(pipeline=hf_pipe)
    print("Langchain Pipeline created")
    return llm_pipeline

def create_chain(prompt, llm_pipeline):
    chain = prompt | llm_pipeline.bind(stop=["USER:","\n\n"])
    return chain

if __name__ == "__main__":
    model_id="togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
    #model_id= "mistralai/Mistral-7B-v0.1"
    chain = create_chain(create_prompt(), 
                         create_llm_pipeline(model_id=model_id,
                                            device_map="cuda:0", load_in_8bit=False, load_in_4bit=True))

    keep_running = True
    while keep_running:
        print("input something...")
        question = input()
        if question != "exit":
            print(chain.invoke({"question": question}))
        else:
            keep_running = False