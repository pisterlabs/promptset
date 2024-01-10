from langchain.chat_models import ChatOpenAI
from langchain.schema import Document, SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

my_document = Document(page_content="this is my document and it is prettylong", 
         metadata={"my_document_id": "1234", "source": "youtube","created_time": "2021-01-01",})

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
# chat([
#     SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says"),
#     HumanMessage(content="I would like to go to New York, how should I do this?")
# ])
output = chat(messages=[
    SystemMessage(content="You are an helpful AI bot"),
    HumanMessage(content="What’s the weather like in Toronto (Canada) right now?")
    ],
    functions=[
        {
            "name": "weather_and_location",
            "description": "This function will return the weather and location of a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "the city and the state, e.g. Toronto, Ontario"
                    },
                    "weather": {
                        "type": "string",
                        "enum": ["celsius", "farhenheit"]
                    }
                },
                "required": ["location", "weather"]
            }
        }
    ]
)

print(output)