from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

def create_prompt():
    template = """You are a question answering Large Language Model.
    Answer the following question.
    USER: {question}
    ASSISTANT:"""
    prompt = PromptTemplate.from_template(template)
    return prompt

def create_llm_pipeline(model_id):
    llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    device=0,  # -1 for CPU
    batch_size=1,  
    model_kwargs={"do_sample": True, "max_length": 128},
    )
    return llm_pipeline

def create_chain(prompt, llm_pipeline):
    chain = prompt | llm_pipeline.bind(stop=["USER:"])
    return chain

if __name__ == "__main__":
    model_id = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
    #model_id = "mistralai/Mistral-7B-v0.1"
    chain = create_chain(create_prompt(), create_llm_pipeline(model_id=model_id))

    keep_running = True
    while keep_running:
        print("input something...")
        question = input()
        if question != "exit":
            print(chain.invoke({"question": question}))
        else:
            keep_running = False