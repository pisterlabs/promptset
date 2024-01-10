from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
import os


openai_key = os.environ.get("OPEN_API_KEY")

def calorie_count(user_data: str )->float:
    prompt = ChatPromptTemplate.from_template("tell me the total calorie count of all food combined , just use approximations no need to be accurate{foo}")
    model_name = "gpt-4-0613"
    model = ChatOpenAI(api_key=openai_key,model=model_name)
    chain = prompt | model

    output = chain.invoke({"foo":user_data })
    prompt = ChatPromptTemplate.from_template("tell me largest number in this sentence , the output should be just a number {foo}")
    functions = [
        {
            "name": "joke",
            "description": "A number",
            "parameters": {
                "type": "object",
                "properties": {
                    "setup": {"type": "integer", "description": "tell me largest number in this sentence"},
                    
                },
                "required": ["setup"],
            },
        }
    ]
    chain = (
        prompt
        | model.bind(function_call={"name": "joke"}, functions=functions)
        | JsonKeyOutputFunctionsParser(key_name="setup")
    )

    output2 = chain.invoke({"foo":output})
    #print(output)
    #print(output2)
    return output2
def main():
    input1 = "1 banan"
    output = calorie_count(input1)

    print(f" the calorie count of {input1} is {output}")
if __name__ == "__main__":
    main()