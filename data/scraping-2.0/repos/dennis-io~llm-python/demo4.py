# Import necessary modules
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize a HuggingFaceHub object with the T5 model fine-tuned on WikiSQL
hub_llm = HuggingFaceHub(
    repo_id="mrm8488/t5-base-finetuned-wikiSQL"
)

# Define a prompt template with an input variable "question"
prompt = PromptTemplate(
    input_variables=["question"],
    template = "Translate English to SQL: {question}"
)

# Initialize an LLMChain object with the prompt template and HuggingFaceHub object
hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

# Generate a response for the input question "What are the top 3 most used devices where users sign up with?" and print it
print(hub_chain.run("What are the top 3 most used devices where users sign up with?"))
