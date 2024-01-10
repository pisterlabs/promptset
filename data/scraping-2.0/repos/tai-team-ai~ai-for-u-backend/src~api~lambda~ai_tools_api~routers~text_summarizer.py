import logging
import os
from enum import Enum
from pathlib import Path
import sys
import threading
from typing import Optional
from fastapi import APIRouter, Response, status, Request
import openai

from pydantic import conint, Field


sys.path.append(Path(__file__, "../").absolute())
from gpt_turbo import GPTTurboChatSession, GPTTurboChat, Role, get_gpt_turbo_response
from utils import (
    AIToolModel,
    BaseAIInstructionModel,
    UUID_HEADER_NAME,
    update_user_token_count,
    sanitize_string,
    EXAMPLES_ENDPOINT_POSTFIX,
    ExamplesResponse,
    BASE_USER_PROMPT_PREFIX,
    error_responses,
    TOKENS_EXHAUSTED_LOGIN_JSON_RESPONSE,
    TOKENS_EXHAUSTED_FOR_DAY_JSON_RESPONSE,
    TokensExhaustedException,
    AIToolResponse,
    append_field_prompts_to_prompt,
)
from text_examples import CEO_EMAIL, ARTICLE_EXAMPLE, CONTRACT_EXAMPLE, TRANSCRIPT_EXAMPLE

router = APIRouter()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


MAX_TOKENS_FROM_GPT_RESPONSE = 400

ENDPOINT_NAME = "text-summarizer"


class SummarySectionLength(Enum):
    SHORT = 'short'
    MEDIUM = 'medium'
    LONG =  'long'


class TextSummarizerInstructions(BaseAIInstructionModel):
    length_of_summary_section: Optional[SummarySectionLength] = Field(
        default=SummarySectionLength.MEDIUM,
        title="Length of Summary Section",
        description="This controls the relative length of the summary paragraph.",
    )
    bullet_points_section: Optional[bool] = Field(
        default=False,
        title="Include Bullet Points Section",
        description="Whether or not to include a bullet points section in the response.",
    )
    action_items_section: Optional[bool] = Field(
        default=False,
        title="Include Action Items Section",
        description="Whether or not to include a bullet points section in the response. Action items are often applicable for meeting notes, lectures, etc.",
    )

SYSTEM_PROMPT = (
    "You are an expert at writing cover letters. You have spent hours "
    "perfecting your cover letter writing skills. You have written cover "
    "letters for hundreds of people. Because of your expertise, I want you "
    "to write a cover letter for me. You should ONLY respond with the "
    "cover letter and nothing else. I will provide you with a resume, job "
    "posting, and optionally a company name to write a cover letter for. You "
    "should respond with a cover letter that is tailored to the job posting "
    "and company, highlights my skills, and demonstrates enthusiasm for "
    "the company and role. I may also ask you to specifically highlight "
    "certain skills from my resume that i feel are most relevant to the job "
    "posting. It is important that you highlight these skills if I ask you to. "
    "Remember, you are an expert at writing cover letters. I trust you to "
    "write a cover letter that will get me the job. Please do not respond with "
    "anything other than the cover letter."
)

valid_summary_lengths = ", ".join([section.value for section in SummarySectionLength])

SYSTEM_PROMPT = (
    "You are an expert at summarizing text. You have spent hours "
    "perfecting your summarization skills. You have summarized text for "
    "hundreds of people. Because of your expertise, I want you to summarize "
    "text for me. You should ONLY respond with the summary in markdown format "
    "and nothing else. I will provide you with a text to summarize. You "
    "should respond with a summary that is tailored to the text, highlights "
    "the most important points, and writes from the same perspective as the "
    "writer of the text. I will specify how long this summary should be by specifying "
    f"it's length with the following options: {valid_summary_lengths}. You should ensure "
    "that you keep this length in mind when summarizing the text. If I ask you to include "
    "bullet points or action items, please use a minimum of 3 bullet points or action items "
    "unless you feel that less is appropriate. Remember, you are an expert at summarizing "
    "text. I trust you to summarize text that will be useful to me. Please do "
    "not respond with anything other than the summary in markdown format with "
    "each section header in bold."
)

class TextSummarizerRequest(TextSummarizerInstructions):
    text_to_summarize: str = Field(
        ...,
        title="Text to Summarize",
        description="The text that you wanted summarized. (e.g. articles, notes, transcripts, etc.)",
    )

class TextSummarizerExampleResponse(ExamplesResponse):
    examples: list[TextSummarizerRequest]
    
    
@router.get(f"/{ENDPOINT_NAME}-{EXAMPLES_ENDPOINT_POSTFIX}", response_model=TextSummarizerExampleResponse, status_code=status.HTTP_200_OK)
async def sandbox_chatgpt_examples() -> TextSummarizerExampleResponse:
    """Return examples for the text summarizer endpoint."""
    examples = [
        TextSummarizerRequest(
            text_to_summarize=CONTRACT_EXAMPLE,
            length_of_summary_section=SummarySectionLength.SHORT,
            bullet_points_section=True,
            action_items_section=True,
        ),
        TextSummarizerRequest(
            text_to_summarize=TRANSCRIPT_EXAMPLE,
            length_of_summary_section=SummarySectionLength.SHORT,
            bullet_points_section=True,
            action_items_section=False,
        ),
        TextSummarizerRequest(
            text_to_summarize=ARTICLE_EXAMPLE,
            length_of_summary_section=SummarySectionLength.MEDIUM,
            bullet_points_section=True,
            action_items_section=True,
        ),
        TextSummarizerRequest(
            text_to_summarize=CEO_EMAIL,
            length_of_summary_section=SummarySectionLength.SHORT,
            bullet_points_section=True,
            action_items_section=True,
        ),
    ]
    response = TextSummarizerExampleResponse(
        example_names=["Contract", "Transcript", "Article", "Email"],
        examples=examples
    )
    return response

@router.post(f"/{ENDPOINT_NAME}", response_model=AIToolResponse, responses=error_responses)
async def text_summarizer(text_summarizer_request: TextSummarizerRequest, request: Request):
    """**Summarize text using GPT-3.**"""
    logger.info(f"Received request: {text_summarizer_request}")
    user_prompt = append_field_prompts_to_prompt(
        TextSummarizerInstructions(**text_summarizer_request.dict()),
        BASE_USER_PROMPT_PREFIX,
    )
    user_prompt += f"\nHere's the text that i want you to summarize for me:\n{text_summarizer_request.text_to_summarize}"
    uuid = request.headers.get(UUID_HEADER_NAME)
    user_chat = GPTTurboChat(
        role=Role.USER,
        content=user_prompt
    )
    try:
        chat_session = get_gpt_turbo_response(
            system_prompt=SYSTEM_PROMPT,
            chat_session=GPTTurboChatSession(messages=[user_chat]),
            frequency_penalty=0.6,
            presence_penalty=0.5,
            temperature=0.3,
            uuid=uuid,
            max_tokens=MAX_TOKENS_FROM_GPT_RESPONSE
        )
    except TokensExhaustedException as e:
        if e.login:
            return TOKENS_EXHAUSTED_LOGIN_JSON_RESPONSE
        return TOKENS_EXHAUSTED_FOR_DAY_JSON_RESPONSE

    latest_gpt_chat_model = chat_session.messages[-1]
    update_user_token_count(uuid, latest_gpt_chat_model.token_count)
    latest_chat = latest_gpt_chat_model.content
    latest_chat = sanitize_string(latest_chat)

    response_model = AIToolResponse(
        response=latest_chat,
    )
    logger.info(f"Returning response for {ENDPOINT_NAME} endpoint.")
    return response_model
