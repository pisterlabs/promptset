from langchain import HuggingFaceHub, LLMChain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

hub_llm = HuggingFaceHub(
    repo_id="gpt2",
    model_kwargs={"max_length": 100},
  )

prompt = PromptTemplate(
    input_variables=["question"],
    template="Give me the answer to the following sports question: {question}",
)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
print(hub_chain.run("Who won world cup in 1998?"))
