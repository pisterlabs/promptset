from config import get_OpenAI
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

#import Pydantic
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator, validator
from typing import List

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

email: {email}
{format_instructions}

"""

# Define Class
class ErrorInfo(BaseModel):
    num_incidents: int = Field(description="this is an integar or the number of errors")
    user_id : str = Field(description="This is the customers user id")
    import_id : str = Field(description="This is the Import Id")
    
    # Custom validator
    @field_validator("num_incidents")
    @classmethod
    def num_incidents_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("number of incidents must be positive")
        return v

# set up the parser
pydantic_parser = PydanticOutputParser(pydantic_object=ErrorInfo)
format_instructions = pydantic_parser.get_format_instructions()


update_template = ChatPromptTemplate.from_template(template=email_template_revised)
messages = update_template.format_messages(email=email_response, format_instructions=format_instructions)

format_response = chat(messages)
print(type(format_response.content))

error_object = pydantic_parser.parse(format_response.content)

print(error_object)
print(type(error_object))