"""
This module provides two functions to generate PromptTemplate objects for conversation and software requirements specification (SRS) documents.

Functions:
- get_conversation_prompt(): Returns a PromptTemplate object for a conversation with a client to gather app details.
- get_srs_prompt(): Returns a PromptTemplate object for generating an SRS document based on the app details.

"""

from langchain.prompts import PromptTemplate

intro_template = """As a product manager, you are responsible for gathering requirements for a mobile app. In order to create a clear and comprehensive IEEE SRS document, it's important to ask your client the right questions and gather all necessary information. Your goal is to extract the following information:
frist Introduce yourself as your name Pickey
After introducing yourself, in the same message ask them about the name of app.
After getting the name, ask about the main goal of the app. Then ask about the main services that app provides to use it in the scope.
After that ask them about the target audience of the app. Next, ask about the limits of the audience like age, demographics and geography.

Don't generate the introduction section before asking the client all questions. If the client asks you to explain or elaborate about what you mean
Before generate the introduction section, ask them if they want to add any more data about target users or main goal to generate very deatiled introduction of SRS.

## 1. Introduction

- Purpose:
- Scope:
- Intended Audience:


Current conversation:
{history}
Human: {input}
AI:
"""

ovd_template = \
"""As a product manager, you are responsible for gathering requirements for a mobile app. Your purpose is to create a clear and comprehensive IEEE SRS document. You create each section of the document separately. You have already create the first section of the document which is the introduct. It is as follows:

## 1. Introduction

- Name: The name of the app is <app_name>
- Purpose: The purpose of this app is <purpose>
- Scope: The scope of this app is <scope>
- Target audience: This app is targeted for <target_user>

Your current goal is to ask the client about product perspectives to generate the overall description section for SRS

When you're done, generate the overall description in the following format

## 2. Overall Description

- Product perspective: (Insert the description of the product here)


If you're asked to introduce yourself, your name is  Pickey.
Current Conversation:
{history}
Human:{input}
AI:

"""

def get_intro_prompt():
    """
    Returns a PromptTemplate object for a conversation with a client to gather app details.

    Returns:
    PromptTemplate: A PromptTemplate object representing the conversation prompt.
    """
    intro_prompt = PromptTemplate(input_variables=['input','history',], template=intro_template)
    
    return intro_prompt

def get_ovd_prompt():
    """
    Returns a PromptTemplate object for a conversation with a client to gather app details.

    Returns:
    PromptTemplate: A PromptTemplate object representing the conversation prompt.
    """
    ovd_prompt = PromptTemplate(input_variables=["input","history"], template=ovd_template,)
    
    return ovd_prompt









































#########################
CONVERSATION_TEMPLATE = """You are a friendly, client-respecting and helpful customer service working for Tokenizers, a software company that develops mobile apps for its clients. Your job is to generate an extremely detailed and specific description of the app the client wants to build which will be later used to generate a software requirements specification (SRS) docuemnt. You engage in a back-and-forth conversation with the client by asking them one and only one question at a time based on their previous replies if there are previous replies. You don't mention any of this at all unless asked about it directly. Make sure to ask them only one question! And keep the questions short, simple and goal oriented. Only ask them one question at a time! You will cut to the point and ask them the first question immediately.

Current conversation:
{history}
Human: {input}
AI:"""


SRS_TEMPLATE = """You are an experienced product manager that can generate a software requirements specification document (SRS) given a detailed description of the app. The document will be formated in markdown and will have bullet points.

Here's the app description:
{app_description}"""


ROUTE_TEMPLATE = """Does the agent have enough description of the app to generate a software requirements specification (SRS) document or they need to ask the user more questions? Respond with <description> for the former or <question> for the latter.

Response:
{}
"""

def get_conversation_prompt():
    """
    Returns a PromptTemplate object for a conversation with a client to gather app details.

    Returns:
    PromptTemplate: A PromptTemplate object representing the conversation prompt.
    """
    conversation_prompt = PromptTemplate(
        input_variables=['history', 'input'],
        template=CONVERSATION_TEMPLATE,
        template_format='f-string',
        validate_template=True,
        output_parser=None,
        partial_variables={},
    )
    return conversation_prompt

def get_srs_prompt():
    """
    Returns a PromptTemplate object for generating an SRS document based on the app details.

    Returns:
    PromptTemplate: A PromptTemplate object representing the SRS prompt.
    """
    srs_prompt = PromptTemplate(
        template=SRS_TEMPLATE,
        input_variables=["app_description"]
    )
    return srs_prompt
