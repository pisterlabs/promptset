from langchain.chains import LLMChain
# from langchain.llms import OpenAI
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
)

response = client.images.generate(
    model = "dall-e-3",
    prompt = "Create image of a cat",
    size = "1024x1024",
    quality = "standard",
    n = 1
)

image_url = response.data[0].url

print(image_url)