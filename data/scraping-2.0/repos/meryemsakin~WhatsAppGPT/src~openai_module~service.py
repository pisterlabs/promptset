

import openai
from typing import TypeVar, Type
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
  HumanMessage,
)
from langchain.prompts import (
  SystemMessagePromptTemplate, 
  ChatPromptTemplate,
  PromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import (
  AIMessage,
  HumanMessage,
)
from pydantic import BaseModel

from src.config import config
from src.user.models import UserModel
from .schemas import ChatCompletionMessageSchema, ChatCompletionOptionsSchema, ChatCompletionResponseSchema
import src.openai_module.prompts as prompts

openai.api_key = config.OPENAI_API_KEY

def create_full_chat_completion(
  message_history: list[AIMessage | HumanMessage],
  prompt: str,
  additional_data: dict = {}
) -> str:
  try:
    chat_openai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    system_message_prompt = SystemMessagePromptTemplate.from_template(
      prompt
    )

    messages = [
      system_message_prompt,
      *message_history
    ]

    chat_prompt_template = ChatPromptTemplate.from_messages(messages)

    chat_prompt = chat_prompt_template.format_messages(**additional_data)

    chat_result = chat_openai.predict_messages(chat_prompt)

    return chat_result
  except Exception as e:
    print("Error: ", e)
    return ""

def create_chat_message(message: ChatCompletionMessageSchema) -> HumanMessage | AIMessage | None:
  if message.role == "assistant":
    return AIMessage(
      content=message.content
    )
  
  if message.role == "user":
    return HumanMessage(
      content=message.content
    )
  
  return None

def create_chat_completion(messages: list[ChatCompletionMessageSchema], options: ChatCompletionOptionsSchema) -> ChatCompletionResponseSchema:
  chat_completion = openai.ChatCompletion.create(
    messages=[message.dict() for message in messages],
    **options.dict()
  )

  return chat_completion

T = TypeVar("T", bound=BaseModel)

def parse_chat_history(
  chat_history: list[AIMessage | HumanMessage], 
  pydantic_model: Type[T], 
  system_prompt_template: str = ""
) -> T:
  try:
    llm = OpenAI(
      temperature=0.0
    )
    
    chat_parser = PydanticOutputParser(
      pydantic_object=pydantic_model
    )

    chat_parser_prompt = PromptTemplate.from_template(system_prompt_template)

    stringified_chat_history = chat_stringify(chat_history)

    prompt = chat_parser_prompt.format(chat_history=stringified_chat_history, format_instructions=chat_parser.get_format_instructions())

    parse_result = llm.predict(prompt)

    parsed = chat_parser.parse(parse_result)

    return parsed
  except Exception as e:
    print("E", e)

def chat_stringify(chat_history: list[AIMessage | HumanMessage]) -> str:
  chat_history_string = ""

  for message in chat_history:
    if isinstance(message, AIMessage):
      chat_history_string += f"AI: {message.content}\n"
      continue

    if isinstance(message, HumanMessage):
      chat_history_string += f"User: {message.content}"

  return chat_history_string

# def create_full_chat_completion(
#   message_history: list[ChatCompletionMessageSchema],
#   user: UserModel, 
#   options: ChatCompletionOptionsSchema
# ) -> ChatCompletionMessageSchema:
#   finish_reason = ""
#   chat_completion_text_result = ""
  
#   user_bio = user.bio_information.dict() if user.bio_information else {}

#   system_message = ChatCompletionMessageSchema(
#     role="system",
#     content=f"""
#       Your name is Serene, An AI Assistant that help me find my ideal partner, you will ask questions about my preference and criteria,
#       With your extensive knowledge of compatibility and your warmth, empathetic nature, Serene is passionate for creating meaningful connections.

#       This is what you know about me:
#       1. Full Name: {user_bio.get("full_name", "")}
#       2. Date Of Birth: {user_bio.get("date_of_birth", "")}
#       3. Gender: {user_bio.get("gender", "")}
#       4. Interests: {user_bio.get("interests", [])}
#       5. Relationship Goal: {user_bio.get("relationship+goal", "")}

#       You will ask questions to fill the fields with empty value, 
#       skip questions for already filled in value.
      
#       Don't ask too many details and ask one question at a time, If the answer doesn't make sense
#       Try asking again and suggests a proper format.
#     """,
#     name="Serene"    
#   )

#   chat_completion = None

#   try:
#     while finish_reason in ["", "length"]:
#       messages = [
#         system_message,
#         *message_history,
#       ]

#       if finish_reason == "length":
#         continue_message = ChatCompletionMessageSchema(
#           role="user",
#           content="continue",
#           name=user.name.split(" ")[0]
#         )

#         messages.append(continue_message)

#       chat_completion: ChatCompletionResponseSchema = openai.ChatCompletion.create(
#         messages=[message.dict() for message in messages],
#         **options.dict()
#       )

#       finish_reason = chat_completion.choices[0].finish_reason
#       chat_completion_text_result = chat_completion_text_result + chat_completion.choices[0].message.content
    

#     if chat_completion:
#       chat_completion.choices[0].message.content = chat_completion_text_result
    
#     return chat_completion
  
#   except RateLimitError as e:
#     print("e", e)
    
#     return None
#   except Exception as e:
#     print("e", e)
    
#     return None
#   finally:
#     return chat_completion
    