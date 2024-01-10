from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm=ChatGoogleGenerativeAI(model="gemini-pro-vision",google_api_key="apikey")

message=HumanMessage(
    content=[
        {
            "type":"text",
            "text":"Describe the image in a single sentance?",
        },
        {
            "type":"image_url",
            "image_url":"https://idsb.tmgrup.com.tr/ly/uploads/images/2022/12/19/247181.jpg"
        }
    ]
)

respose=llm.invoke([message])

print(respose.content)
#  Lionel Messi holds the World Cup trophy after winning the 2022 FIFA World Cup Final.