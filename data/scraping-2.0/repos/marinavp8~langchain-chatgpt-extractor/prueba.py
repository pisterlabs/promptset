from langchain.chat_models import  ChatOpenAI 
from langchain.chains import  create_extraction_chain 


import os 
from dotenv import load_dotenv

load_dotenv()
OPENAI = os.environ.get("OPENAI_API_KEY")

description_text = """
MacBook Pro M2 Max takes its power and efficiency further than ever.
It delivers exceptional performance whether it's plugged in or not, 
and now has even longer battery life. Combined with a stunning 
Liquid Retina XDR display and all the ports you need - this is a pro
laptop without equal.Supercharged by 12-core CPU Up to 38-core GPU
Up to 96GB unified memory 400GB/s memory bandwidth. 
"""

# Complete the schema
schema = {
    "properties": {
        "name": {"type": "string"},
        "ram": {"type": "string"},
        "cpu": {"type": "string"},
        "gpu": {"type": "string"},

    },
    
}

def extract_specs_from_description(description_text):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",
                     openai_api_key=OPENAI)
    chain = create_extraction_chain(schema, llm)
    return chain.run(description_text)

# Call the defined extract_specs_from_description() function, passing the description text
results = extract_specs_from_description(description_text)

# See the results
print(results)
