from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
# llm = ChatGoogleGenerativeAI(model="gemini-pro")

# for chunk in llm.stream("what is a LLM?"):
#     print(chunk.content)
#     print("---")

image_url = "https://picsum.photos/seed/picsum/300/300"

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What color is the sky? is it pink or blue? pick one",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": image_url},
    ]
)
print(llm.invoke([message]))
