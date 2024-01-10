import os
import vertexai

from langchain.llms import VertexAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]

vertexai.init(project=PROJECT_ID, location=REGION)

llm = VertexAI(model_name="gemini-pro")

result = llm.invoke("What are some of the pros and cons of Python as a programming language?")
print(result)

llm = VertexAI(model_name="gemini-pro-vision")

image_url = "mages/fish_market.png"

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Describe this image? Extract any text from the image as well.",
        },
        {"type": "image_url", "image_url": image_url},
    ]
)
result = llm.invoke([message])
print(result)

