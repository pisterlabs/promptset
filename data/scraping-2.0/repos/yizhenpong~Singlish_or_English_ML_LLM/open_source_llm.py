"""
instructions:
- download ollama `https://ollama.ai/download`
    - ollama pull llama2
    - ollama pull mistral
- pip install langchain
"""

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser #PydanticOutputParser
# from langchain_core.pydantic_v1 import BaseModel, Field, validator
import langchain

'''
Run llama2 and get outputs for this large language model
'''

############################################################################################################################## 

'''loading the model'''

llm = Ollama(model='llama2',
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
llm2 = Ollama(model='mistral',
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


'''building schemas'''
sentence_schema = ResponseSchema(name="sentence", description="This is the input sentence")
label_schema = ResponseSchema(name="label", 
        description="""
        Label for the input sentence, which should be an integer label of 0 or 1.
        Please do not give comments after 0 or 1 of the form '//' or '#'
        If you are unable to classify it, provide the reason and classify it under 2""", type="int")

# to run explanation, comment line 42 and uncomment lines 44-46
response_schemas = [sentence_schema, label_schema]

# explanation_schema = ResponseSchema(name="explanation", 
#         description="Reasoning for why the sentence was classified as the label")
# response_schemas = [sentence_schema, label_schema, explanation_schema]


'''getting structured output'''
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

template_string = """ 
    Given this sentence: '{sentence}', classify it as integer 0 or 1. \n
        note that: 0 represents Singlish while 1 represent English
        Here are some format instructions that you must follow:
        {format_instructions} 
        Ensure that all strings are enclosed in double quotes,
        The label you provide must be strictly an integer output of either 0 or 1 with no comments like '//' or '#' beside it,
        Do not write any extra lines beyond the json output
        """
prompt = PromptTemplate(
    template=template_string,
    input_variables=["sentence"],
    partial_variables={"format_instructions": format_instructions},
)

def get_output_dict_llama(sentence):
    try:
        _input = prompt.format_prompt(sentence=sentence)
        # output = llm(_input.to_json)
        output = llm(_input.to_string())
        """ to resolve this error:
            langchain.schema.output_parser.OutputParserException: Got invalid JSON object.
            https://stackoverflow.com/questions/77396803/langchain-schema-output-parser-outputparserexception-got-invalid-json-object-e
        """
        if output.find("\`\`\`\n\`\`\`") != -1:
            output = output.replace("\`\`\`\n\`\`\`", "\`\`\`")
        return output_parser.parse(output) #dict type
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        # Handle other exceptions as needed
        return {"sentence": sentence, "label": 2}


def get_output_dict_mistral(sentence):
    try:
        _input = prompt.format_prompt(sentence=sentence)
        # output = llm(_input.to_json)
        output = llm2(_input.to_string())
        """ to resolve this error:
            langchain.schema.output_parser.OutputParserException: Got invalid JSON object.
            https://stackoverflow.com/questions/77396803/langchain-schema-output-parser-outputparserexception-got-invalid-json-object-e
        """
        if output.find("\`\`\`\n\`\`\`") != -1:
            output = output.replace("\`\`\`\n\`\`\`", "\`\`\`")
        return output_parser.parse(output) #dict type
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        # Handle other exceptions as needed
        return {"sentence": sentence, "label": 2}


# def get_output_dict(sentence):
#    answer=chain.run(sentence)
#    """ to resolve this error:
#         langchain.schema.output_parser.OutputParserException: Got invalid JSON object.
#         https://stackoverflow.com/questions/77396803/langchain-schema-output-parser-outputparserexception-got-invalid-json-object-e
#    """
#    if answer.find("\`\`\`\n\`\`\`") != -1:
#      answer = answer.replace("\`\`\`\n\`\`\`", "\`\`\`")
#      #print("Found and replaced")
#    output_dict = output_parser.parse(answer)
#    return output_dict


############################################################################################################################## 
"""test case"""

# sentence1 = "I already have a host and blogger"
# sentence2 = "Meet after lunch la..."

# x = get_output_dict(sentence1)
# print(x)
# print(type(x))
# print(x.keys())
# print(x['label'])



'''using JSON parser instead'''
# Define your desired data structure.
# class Sentence(BaseModel):
#     sentence: str = Field(description="This is the input sentence")
#     label: int = Field(description="""Integer type is expected, do not put any comments beside it. 
#                     This label must be strictly an integer label of 0 or 1""")
#     explanation: str = Field(description="Reasoning for why the sentence was classified as the label")

#     # # You can add custom validation logic easily with Pydantic.
#     # @validator("sentence")
#     # def question_ends_with_question_mark(cls, field):
#     #     if field[-1] != "?":
#     #         raise ValueError("Badly formed question!")
#     #     return field
    
# template_string = """
#     Given this sentence: '{sentence}', classify if it is Singlish (0) or English (1), 
#     the label you provide must be strictly an integer output of either 0 or 1 with no comments like '//' or '#' beside it,
#         {format_instructions}"""
# parser = PydanticOutputParser(pydantic_object=Sentence)
# prompt = PromptTemplate(
#     template=template_string,
#     input_variables=["sentence"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

# def get_output(sentence):
#     _input = prompt.format_prompt(sentence=sentence)
#     output = llm(_input.to_string())
#     return output

# """test case"""
# _input = prompt.format_prompt(sentence=sentence1)
# output = llm(_input.to_string())
# x = parser.parse(output)
# print(type(x))
# print(x['label'])

'''previously used chain'''
# print(prompt.messages)
# chain = LLMChain(llm=llm, 
#                  prompt=prompt)

# def get_output_dict(sentence):
#    answer=chain.run(sentence)
#    """ to resolve this error:
#         langchain.schema.output_parser.OutputParserException: Got invalid JSON object.
#         https://stackoverflow.com/questions/77396803/langchain-schema-output-parser-outputparserexception-got-invalid-json-object-e
#    """
#    if answer.find("\`\`\`\n\`\`\`") != -1:
#      answer = answer.replace("\`\`\`\n\`\`\`", "\`\`\`")
#      #print("Found and replaced")
#    output_dict = output_parser.parse(answer)
#    return output_dict

# def get_output_label(sentence):
#    return get_output_dict(sentence)['label']


# """Create key, value pairs from the following data and format the key, value pairs using JSON notation with keys: 
#             "sentence", "label", "explanation" """


'''try out this strict output
https://www.youtube.com/watch?v=A6sIh-lmApk 
'''
