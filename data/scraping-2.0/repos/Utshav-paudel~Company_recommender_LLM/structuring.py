#@ Script for converting Text unstructured company data to score based on information

## Structure output parser
# importing dependecies
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

def company_score_generator(input_text):
    # designing response schema for our ranking score of company
    response_schemas = [ResponseSchema(name="company_name", description="name of the company" ),
                   ResponseSchema(name="Risk_averse_score", description="score of risk averse of company" ),
                   ResponseSchema(name ="Budget_focus_score", description="score of how budget focus company is " ),
                   ResponseSchema(name="Advanced_score", description="score of how advanced company is with current upto date techs and trends" )
                   ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # prompt to model
    prompt = PromptTemplate(
        template="You are a world class algorithm for giving score to text data ranging from 0 for wrost performer to 10 for best performer \n{format_instructions} \n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    # model  and its input output
    model = OpenAI()

    # text data of company
    # question =
    _input = prompt.format_prompt(question= input_text)
    output = model(_input.to_string())

    result = output_parser.parse(output)
    return result

if __name__=="__main__":
    company_score_generator()

