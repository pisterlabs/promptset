"""
Module for the catchy title creator endpoint.

This endpoint uses the OpenAI API to generate a catchy title for a given text 
provided by the user. The endpoint takes a text, target audience for title, and a number of titles to generate. The endpoint
then constructs a prompt with this data and sends it to the OpenAI API. The response from the API is then returned to the 
client. 

Attributes:
    router (APIRouter): Router for the catchy title creator endpoint.
    CatchyTitleCreatorModel (Pydantic Model): Pydantic model for the request body.
    CatchyTitleCreatorResponseModel (Pydantic Model): Pydantic model for the response body.
    get_openai_response (function): Method to get response from openai.
    catchy_title_creator (function): Post endpoint for the lambda function.
"""
from pydantic import constr, conint, Field
from fastapi import APIRouter, Response, status, Request
from typing import Optional, List
from pathlib import Path
import logging
import sys
import os

sys.path.append(Path(__file__).parent / "../utils")
sys.path.append(Path(__file__).parent / "../text_examples")
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../gpt_turbo"))
from gpt_turbo import GPTTurboChatSession, GPTTurboChat, Role, get_gpt_turbo_response
from text_examples import TEXT_BOOK_EXAMPLE, COFFEE_SHOP, SHORT_STORY
from utils import (
    map_value_between_range,
    AIToolModel,
    sanitize_string,
    BaseAIInstructionModel,
    Tone,
    EXAMPLES_ENDPOINT_POSTFIX,
    docstring_parameter,
    ExamplesResponse,
    AIToolsEndpointName,
    UUID_HEADER_NAME,
    append_field_prompts_to_prompt,
    BASE_USER_PROMPT_PREFIX,
    error_responses,
    TOKENS_EXHAUSTED_LOGIN_JSON_RESPONSE,
    TOKENS_EXHAUSTED_FOR_DAY_JSON_RESPONSE,
    TokensExhaustedException,
    AIToolResponse,
)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

router = APIRouter()

ENDPOINT_NAME = AIToolsEndpointName.CATCHY_TITLE_CREATOR.value
MAX_TOKENS_FROM_GPT_RESPONSE = 200

AI_PURPOSE = " ".join(ENDPOINT_NAME.split("-")).lower()

class CatchyTitleCreatorInstructions(BaseAIInstructionModel):
    type_of_title: Optional[constr(min_length=1, max_length=50)] = Field(
        ...,
        title="What's the Titles/Names For?",
        description="This can literally be anything. (eg. company, coffee shop, song, documentary, social media post, etc.)"
    )
    target_audience: Optional[constr(min_length=1, max_length=200)] = Field(
        ...,
        title="Target Audience",
        description="The target audience for the title (e.g. children, adults, teenagers, public, superiors, etc.)"
    )
    tone: Optional[Tone] = Field(
        default=Tone.INFORMAL,
        title="Tone of the Titles/Names",
        description="The expected tone of the generated titles."
    )
    specific_keywords_to_include: Optional[List[constr(min_length=0, max_length=20)]] = Field(
        default=[],
        title="Keywords to Include",
        description="These can help your title perform better for SEO (e.g. 'how to', 'best', 'top', 'ultimate', 'ultimate guide', etc.)."
    )
    num_titles: Optional[conint(ge=1, le=15)] = Field(
        default=3,
        title="Number of Titles/Names to Generate",
        description="The number of titles that you want to generate."
    )
    creativity: Optional[conint(ge=0, le=100)] = Field(
        65,
        title="Creativity (0 = Least Creative, 100 = Most Creative)",
        description="The creativity of the titles. More creativity may be more inspiring but less accurate while less creativity may be more accurate but less inspiring."
    )

SYSTEM_PROMPT = (
    "You are an expert at creating catchy titles and names for things. I will provide a description of the thing that I want you to create a name for OR "
    "I will provide you a text that you should create a catchy title for. You should infer from context whether I am asking you to create a name or a title. "
    "You should respond with names/catchy titles and nothing else. You should just include titles, no sub titles or descriptions."
    "You should use markdown format when returning the titles and should return them as a list."
    "You should follow the instructions that I provide you in order to generate the best titles and names for me."
    "These instructions could include any of the following:\n"
    "Type of Title: This is what defines the thing you are creating a name or title for. (eg. company, coffee shop, song, documentary, social media post, etc.)\n"
    "Target Audience: You should keep this audience in mind when you are writing the names and titles. Here are some examples: children, adults, teenagers, public, superiors, etc.\n"
    f"Tone: The expected tone of the titles and the names that you generate. Here are the possible tones: {[tone.value for tone in Tone]}.\n"
    "Specific Keywords to Include: You should try to include these keywords in the titles and names that you generate.\n"
    "Number of Titles: The number of name & titles that you should generate.\n"
    "Creativity: The creativity of the names and titles that you create. Where 0 is the least creative and 100 is the most creative.\n"
    "Text or Description: The text or description of what you are generating catchy titles for. For generating titles for text (e.g. articles, blogs, social media posts, songs, etc.), "
        "this should be the text. For other types of things, (e.g. coffee shop, company name, street name etc.) this should be a description of the thing you are generating a name for.\n"
)

class CatchyTitleCreatorRequest(CatchyTitleCreatorInstructions):
    """
    **Define the model for the request body for {0} endpoint.**
    
    **Atrributes:**
    - text: The text to generate a catchy title for.

    **AI Instructions:**

    """

    __doc__ += BaseAIInstructionModel.__doc__
    text_or_description: constr(min_length=1, max_length=10000) = Field(
        ...,
        title="Text or Description",
        description="This can be the text you are generating titles for (article, notes, etc.), or if you are generating names for something (pet, company name, etc.), you can describe what the name is for."
    )



class CatchyTitleCreatorExamplesResponse(ExamplesResponse):
    """
    **Define examples for teh {0} endpoint.**
    
    **Atrributes:**
    - examples: List of examples for the {0} endpoint. Can post these examples to the {0} endpoint without
        modification.
    
    Inherit from ExamplesResponse:
    """
    __doc__ += ExamplesResponse.__doc__
    examples: list[CatchyTitleCreatorRequest]

@docstring_parameter(ENDPOINT_NAME)
@router.get(f"/{ENDPOINT_NAME}-{EXAMPLES_ENDPOINT_POSTFIX}", response_model=CatchyTitleCreatorExamplesResponse, status_code=status.HTTP_200_OK)
async def catchy_title_creator_examples():
    """
    **Get examples for the {0} endpoint.**
    
    This method returns a list of examples for the {0} endpoint. These examples can be posted to the {0} endpoint 
    without modification.
    """
    catchy_title_examples = [
        CatchyTitleCreatorRequest(
            text_or_description=TEXT_BOOK_EXAMPLE,
            target_audience="Young Adults",
            tone=Tone.FRIENDLY,
            num_titles=8,
            creativity=30,
            specific_keywords_to_include=["furry friends"],
            type_of_title="Textbook",
        ),
        CatchyTitleCreatorRequest(
            text_or_description=COFFEE_SHOP,
            target_audience="Travelers & tourists",
            tone=Tone.OPTIMISTIC,
            num_titles=5,
            creativity=75,
            specific_keywords_to_include=["quaint", "cozy", "cute"],
            type_of_title="Coffee Shop",
        ),
        CatchyTitleCreatorRequest(
            text_or_description=SHORT_STORY,
            target_audience="Kids",
            tone=Tone.FRIENDLY,
            num_titles=5,
            creativity=90,
            type_of_title="Short Story",
        ),
    ]
    
    example_response = CatchyTitleCreatorExamplesResponse(
        example_names=["Textbook", "Coffee Shop", "Short Story"],
        examples=catchy_title_examples
    )
    return example_response


@router.post(f"/{ENDPOINT_NAME}", response_model=AIToolResponse, responses=error_responses)
async def catchy_title_creator(catchy_title_creator_request: CatchyTitleCreatorRequest, response: Response, request: Request):
    """**Generate catchy titles using GPT-3.**"""
    logger.info(f"Received request for {ENDPOINT_NAME} endpoint.")
    user_prompt = append_field_prompts_to_prompt(CatchyTitleCreatorInstructions(**catchy_title_creator_request.dict()), BASE_USER_PROMPT_PREFIX)

    user_prompt += f"\nHere is the text/description of what you should create a name or title for: {catchy_title_creator_request.text_or_description}"
    uuid = request.headers.get(UUID_HEADER_NAME)
    user_chat = GPTTurboChat(
        role=Role.USER,
        content=user_prompt,
    )
    logger.info("User prompt: %s", user_prompt)
    temperature = map_value_between_range(catchy_title_creator_request.creativity, 0, 100, 0.2, 1.0)
    presence_penalty = map_value_between_range(catchy_title_creator_request.creativity, 0, 100, 0.3, 1.3)
    frequency_penalty = map_value_between_range(catchy_title_creator_request.creativity, 0, 100, 0.8, 2.0)
    try:
        chat_session = get_gpt_turbo_response(
            system_prompt=SYSTEM_PROMPT,
            chat_session=GPTTurboChatSession(messages=[user_chat]),
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            uuid=uuid,
            max_tokens=MAX_TOKENS_FROM_GPT_RESPONSE,
        )
    except TokensExhaustedException as e:
        if e.login:
            return TOKENS_EXHAUSTED_LOGIN_JSON_RESPONSE
        return TOKENS_EXHAUSTED_FOR_DAY_JSON_RESPONSE

    latest_gpt_chat_model = chat_session.messages[-1]
    latest_chat = latest_gpt_chat_model.content
    logger.info("Latest chat: %s", latest_chat)
    latest_chat = sanitize_string(latest_chat)

    response_model = AIToolResponse(response=latest_chat)
    logger.info("Returning response: %s", response_model)
    return response_model
