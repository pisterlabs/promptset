from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
import dotenv
import os
import boto3

dotenv.load_dotenv()    

# client = boto3.client('dynamodb')
# response = client.get_item(TableName='Garbage_collector_table', Key={'topic':{'S':str(my_topic)}})


# a chain that extracts the event name, date, location and additional details from a text
# a function that takes in a text and returns a list of dictionaries containing the extracted information

def extract_event(context=None,test=False): 

    """Extracts event name, date, location and additional details from a text"""

    extract_schema = {
        "properties": {
            "event_name": {"type": "string",
            "description": "Use title of the event. If title not present, use short decription of the event as the title of the event."
            },
            "date": {"type": "integer"
            },
            "location": {"type": "string"},
              "additional_details": {"type":"string"}
        },
        "required": ["event_name"],
    }

    # Input 

    # Run chain
    llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), max_tokens = 2056,
    temperature=0, model="gpt-3.5-turbo")
    chain = create_extraction_chain(extract_schema, llm)
    response = chain.run(context)

    # Schema
    tag_schema = {
        "properties": {
            "experience_type": {
                "type": "integer",
                "enum": [1, 2, 3, 4, 5],
                "description": "describes the mood of the event. 1 is very calming and 5 is very exciting.",
            },
            "event_type": {
                "type": "string",
                "enum": ["performing_arts", 
                "outdoor and recreation", 
                "cultural and educational", 
                "sports and games", 
                "food and drink",
                "fashion and lifestyle",
                "science and technology",
                "religious, political and civic"
                ],
            },
        },
        "required": ["experience_type","event_type"],
    }

    chain = create_tagging_chain(tag_schema, llm)
    l=[]
    for r in response:
        res = chain.run(str(r))
        r.update(res)
        l.append(r)
    return {"response":l}

############################################################################################################
##Testing##

for i in [e for e in os.listdir('../data/') if 'example' in e]:
    print(i)
    with open(f"../data/{i}", "r") as f:
        context= f.read()
    print(extract_event(context,test=True))
