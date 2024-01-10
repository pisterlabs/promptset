from OpenAI_Training.config import get_OpenAI
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


llm_model = "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0, model=llm_model)

email_response = """
We have an issue with the import of the following customer feed file. There are 5 other customers with the same issue.
Import ID      2289581
Import URL     https://transport.productsup.io/bee95abe02f1f5a18c63/channel/123456/example.xml
User ID        321
Start time     2023-11-28T06:00:13Z
End time     2023-11-28T06:00:13Z
Error          internal error, second error, third error
Internal Error write to spool file: write /feed-downloads/feed-ebayk-2289581: no space left on device

"""

email_template = """
From the following email, please extract the following information:
User_Id: what is the user id?

Import_ID: what is the import id?
start_time: what is the start time?
end_time: what is the end time?
errors: what are the errors? If there are multiple errors, please list them all in square brackets as an array.

Format the response as JSON with the following keys:
    user_id
    import_id
    start_time
    end_time
    errors
    
email = {email}

"""
####################
email_template_revised = """
From the following email, please extract the following information:
User_Id: what is the user id?

Import_ID: what is the import id?
start_time: what is the start time?
end_time: what is the end time?
errors: what are the errors? If there are multiple errors, please list them all in square brackets as an array.

Format the response as JSON with the following keys:
    user_id
    import_id
    start_time
    end_time
    errors
    
email = {email}

"""
# {first_format_instructions}


prompt_template = ChatPromptTemplate.from_template(email_template)


#####################
# Langchain Parsers
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

end_time_schema = ResponseSchema(name="end_time", description="The end time of the import")
start_time_schema = ResponseSchema(name="start_time", description="The start time of the import")
import_id_schema = ResponseSchema(name="import_id", description="The import id of the import")
user_id_schema = ResponseSchema(name="user_id", description="The user id of the import")
errors_schema = ResponseSchema(name="errors", description="The errors of the import")

response_schema = [end_time_schema, start_time_schema, import_id_schema, user_id_schema, errors_schema]

# Create a parser
output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()

updated_prompt_template = ChatPromptTemplate.from_template(template=email_template_revised)
messages = prompt_template.format_messages(email=email_response, format_instructions=format_instructions)
response = chat(messages)

output_dict = output_parser.parse(response.content)

print(type(output_dict))
print(f"User Ids: {output_dict['user_id']}")
print(f"Erorrs: {output_dict['errors']}")




# print(type(reponse.content))
# print(reponse.content)