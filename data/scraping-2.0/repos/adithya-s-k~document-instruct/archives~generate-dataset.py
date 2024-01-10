# import csv

# def print_rows_from_csv(csv_file):
#     with open(csv_file, 'r', encoding='utf-8') as f:
#         reader = csv.reader(f)
        
#         # Skip the header row
#         next(reader)
        
#         # Iterate over each row and print it
#         for row in reader:
#             print("File:", row[0])
#             print("Heading Text:", row[1])
#             print("Content:", row[2])
#             print("Num Tokens:", row[3])
#             print("Num Words:", row[4])
#             print("Generate Data:", row[5])
#             print('-' * 50)  # Separator for clarity

# # Example usage
# csv_file = "output.csv"
# print_rows_from_csv(csv_file)


import csv
import os
import getpass

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import getpass

from rich import print

from langchain.output_parsers import GuardrailsOutputParser

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# specifying the output formate
extractor_rail = """
<rail version="0.1">

<output>
    <list name="" description="">
        <object>
            <string name="instruction" description="" />
            <string name="input" description=""/>
            <string name="output" description="" />
        </object>
    </list>
</output>

<prompt>

given the following instruction list with input and output, please extract a list of different instructions

{{instruction_list}}

@complete_json_suffix_v2
</prompt>
</rail>
"""


chat = ChatOpenAI(temperature=0.9 , model="gpt-3.5-turbo-16k", client=any)

formated_problem = [
    {
        "Questions": "",
        "output": ""
    }
]



def print_first_n_rows_from_csv(csv_file, n=1):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Skip the header row
        next(reader)
        
        # Iterate over each row and print it
        count = 0
        for row in reader:
            Heading = row[1]
            Content = row[2]
            Instruction_number = row[5]
            
            messages = [
                HumanMessage(
                    content=f"""
Given this section of the documentation: {Heading}
Given this code snippet from the documentation in markdown: {Content}
Understand the code snippet and formate into plain python code with comments.

Generate unique instructions whose output is: ```{Content}```
The instructions must be unique and not repeated.

Requirements for the Instructions:
The Instructions should be 1 to 2 sentences long.
The Instructions should focus on a specific code-related task, such as function creation, conditional statements, loops, variable assignments, or any other relevant code operation.
The generated Instructions should be in English and clearly convey the required task related to the code snippet.
"""
                ),
            ]
            instruction_gen = chat(messages).content
            
            print("Generated Instructions: ")
            print(instruction_gen)
            

            print("Parsing Instructions: ")
            output_parser = GuardrailsOutputParser.from_rail_string(extractor_rail)

            prompt = PromptTemplate(
                template=output_parser.guard.base_prompt,
                input_variables=output_parser.guard.prompt.variable_names,
            )

            model = OpenAI(temperature=0) # type: ignore
            parsed_instruction_list = model(prompt.format_prompt(instruction_list=instruction_gen).to_string())
            parsed_instruction_list = output_parser.parse(parsed_instruction_list)
            print(parsed_instruction_list)
            
            print("-"*50)
            count += 1
            if count == n:
                break

# Example usage
csv_file = "output.csv"
print_first_n_rows_from_csv(csv_file)