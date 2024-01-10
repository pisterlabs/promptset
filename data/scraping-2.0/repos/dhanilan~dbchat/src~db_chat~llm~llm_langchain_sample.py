from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI(openai_api_key="sk-xxxxxx")
# chain = prompt | model

functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]
# chain = prompt | model.bind(function_call={"name": "joke"}, functions=functions)


def invoke():
    chain = (
        prompt
        | model.bind(function_call={"name": "joke"}, functions=functions)
        | JsonKeyOutputFunctionsParser(key_name="setup")
    )
    response = chain.invoke({"foo": "bears"})

    print(response)
