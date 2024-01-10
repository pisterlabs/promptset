from langchain.chains import create_extraction_chain, create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from util import initialize

initialize()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

schema = {
    "properties": {
        "person_name": {"type": "string"},
        "person_height": {"type": "integer"},
        "person_hair_color": {"type": "string"},
        "dog_name": {"type": "string"},
        "dog_breed": {"type": "string"},
    },
    "required": ["person_name", "person_height"],
}

inp = """
Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde.
Alex's dog Frosty is a labrador and likes to play hide and seek.
        """
chain = create_extraction_chain(schema, llm)
res = chain.run(inp)
print(res)
