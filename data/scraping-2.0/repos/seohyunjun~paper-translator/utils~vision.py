
# GPT4-Vision 
import base64
from langchain.chat_models import ChatOpenAI

from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def translate_image(image_url: str, max_token: int=512):
    
    chain = ChatOpenAI(model="gpt-4-vision-preview", max_token=1024)


    msg = chain.invoke(
        [   AIMessage(
            content="You are a useful bot that is especially good at OCR from images and explain what you see."
        ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "저것이 무엇인지 한국어로 설명해"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_url}"
                        },
                    },
                ],
                max_tokens=max_token,
            )
        ]
    )
    print(msg.content)
    return msg.content