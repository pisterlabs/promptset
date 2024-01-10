from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

hub_llm=HuggingFaceHub(
    repo_id="gpt2-medium", #llm-model
    model_kwargs={'temperature':0.8,'max_length':100}
)

prompt=PromptTemplate(
    input_variables=["question"],
    template="{question}"
)
hub_chain=LLMChain(prompt=prompt, llm=hub_llm,verbose=True)
print(hub_chain.run("what is the captial of india?"))
