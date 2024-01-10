from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import json
import re


def active_func(query, memory):
    response_schemas = [
        ResponseSchema(name="answer",
                       description="name of function with required or similar means to user's input"),
        ResponseSchema(name="number",
                       description="The ordinal number of the product which is not text that the user desires to try, defaulting to 1.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas)
    previous_action = memory.load_memory_variables(
        {})["history"].split(" ")[-1]
    form = """
                "Given the following user input and a list of functions, identify the function associated with the provided input:

                User Input: [{question}]

                List of Functions:
                1. Description: If users input meaning greeting such as 'hello' or 'hi' or 'hey' or 'good morning' or 'good afternoon' or good evening or good night or how are you or how do you do or how are you doing or how is it going or how is everything or how is life or how is your day or how is your day going or how is your day so far or how is your day been or how is your day been going
                Function: greeting
                
                2. Description: Refuse to answer the question not related to any function.
                Function: refuseToAnswer

                3. Description: Instructions for uploading photos including personal photos and full-body shots.
                Function: uploadPose

                4. Description: Recommend cloth items in database based on user's input.
                Function: recommendCloth

                5. Description: Try or fit or make someone look like in i-th cloth item in user's input, which i is the order of specific cloth that user want.
                Function: tryCloth
                
                6. Description: Users want to repeat again the action or try more or again in try on cloth items.
                Function: [{previous_action}]
                
                7. Description: Users provide their body measurements such as height, weight, age and sex.
                Function: predictSize
                
                8. Description: Users want to know more information about the cloth item in user's input.
                Function: showDetail
                
                9. Description: Users want to delete the user's uploaded photo.
                Function: deletePose
                
                Which function is most closely associated with the provided user input? Choose the corresponding function name from the list."
                {format_instructions}
            """
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        input_variables=["question", "previous_action"],
        template="Answer the following questions as best you can." + form +
        "If you do not know the answer, then return function name: refuseToAnswer.",
        partial_variables={"format_instructions": format_instructions}
    )

    model = OpenAI(temperature=0)
    conversation = ConversationChain(
        llm=model)
    _input = prompt.format_prompt(
        question=query, previous_action=previous_action)
    output = conversation(_input.to_string())

    result = str(output["response"]).replace('`', '').replace(
        'json', '').replace('\t', '').replace(' ', '').replace('\n', '')
    result = json.loads(re.search(r'\{(.+?)\}', result).group(0))
    answer = result["answer"]
    number = result["number"]
    if not result == "refuseToAnswer":
        memory.save_context({"input": query}, {"output": answer})
    return answer, number


if __name__ == "__main__":
    memory = ConversationBufferMemory()
    while True:
        inp = input("Input: ")
        answer, number = active_func(inp, memory)
        print(answer)
        print(number)
