from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())


def main():
    description_text = """
    MacBook Pro M2 Max takes its power and efficiency further than ever.
    It delivers exceptional performance whether it's plugged in or not, 
    and now has even longer battery life. Combined with a stunning 
    Liquid Retina XDR display and all the ports you need - this is a pro
    laptop without equal.Supercharged by 12-core CPU Up to 38-core GPU
    Up to 96GB unified memory 400GB/s memory bandwidth. 
    """

    schema = {
        "properties": {
            "name": {"type": "string"},
            "ram": {"type": "string"},
            "cpu": {"type": "string"},
            "gpu:": {"type": "string"},
            "display": {"type": "string"},
        },
        "required": ["name", "ram", "cpu"],
    }
    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    chain = create_extraction_chain(schema, llm)
    print(chain.run(description_text))


main()

# OUTPUT: {'cpu': '12-core', 'name': 'MacBook Pro M2 Max', 'ram': '96GB'}
