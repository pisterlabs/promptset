# Import the required modules
from langchain.llms import Clarifai
from langchain import PromptTemplate, LLMChain

# Clarifai API key - Personal Access Token
CLARIFAI_PAT = 'c18a2b6b798045fb9d3c6b0dbf9a0f5b'

# Input
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Setup
# Specify the correct user_id/app_id pairings
# Since you're making inferences outside your app's scope
USER_ID = 'meta'
APP_ID = 'Llama-2'
# Change these to whatever model and text URL you want to use
MODEL_ID = 'llama2-13b-chat'
MODEL_VERSION_ID = '79a1af31aa8249a99602fc05687e8f40'

# Initialize a Clarifai LLM
clarifai_llm = Clarifai(
    pat=CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID
)

# Create LLM chain
llm_chain = LLMChain(prompt=prompt, llm=clarifai_llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)