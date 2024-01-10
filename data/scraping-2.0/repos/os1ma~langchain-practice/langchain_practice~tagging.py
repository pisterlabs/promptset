from langchain.chains import create_tagging_chain
from langchain.chat_models import ChatOpenAI
from util import initialize

initialize()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

schema = {
    "properties": {
        "sentiment": {"type": "string"},
        "aggressiveness": {"type": "integer"},
        "language": {"type": "string"},
    }
}

chain = create_tagging_chain(schema, llm)

inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
res = chain.run(inp)
print(res)
