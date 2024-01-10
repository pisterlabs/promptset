from langchain.agents import initialize_agent
from langchain.agents import load_tools
import os
from langchain.llms import AzureOpenAI
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://soumenopenai.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "3a5a6eba4d2546558d3fa749ef9fb5ce"
os.environ["deployment_name"] = "gpt-35-turbo"


llm = AzureOpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
)
# chain = LLMChain(llm=llm, prompt=prompt)
# image_url = DallEAPIWrapper().run(chain.run("halloween night at a haunted museum"))


tools = load_tools(['dalle-image-generator'])
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True)
output = agent.run("Create an image of a halloween night at a haunted museum")

print(output)
