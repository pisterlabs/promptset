# Import necessary modules
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize a HuggingFaceHub object with the GPT-2 model
hub_llm = HuggingFaceHub(
    repo_id="gpt2",
    model_kwargs={'temperature': 0.8, 'max_length': 100}
)

# Define a prompt template with an input variable "profession"
prompt = PromptTemplate(
    input_variables=["profession"],
    template = "You had one job! 😠 You're the {profession} and you didn't have to be sarcastic"
)

# Initialize an LLMChain object with the prompt template and HuggingFaceHub object
hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

# Generate a response for the input "CTO" and print it
print(hub_chain.run("CTO"))

# Generate a response for the input "cancellor" and print it
print(hub_chain.run("cancellor"))

