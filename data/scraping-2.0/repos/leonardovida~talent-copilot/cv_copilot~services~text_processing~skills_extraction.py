import logging

import instructor
from openai import AsyncOpenAI, OpenAI

from cv_copilot.services.llm.models.skills import JobDescriptionExtract
from cv_copilot.services.llm.prompt_templates.cv import system_prompt_cv, user_prompt_cv
from cv_copilot.services.llm.prompt_templates.job_descriptions import (
    system_prompt_job_description,
    user_prompt_job_description,
)
from cv_copilot.settings import settings
from cv_copilot.web.dto.job_description.schema import JobDescriptionModel


async def parse_skills_job_description(
    job_description: JobDescriptionModel,
) -> JobDescriptionExtract:
    """Parse the skills from the job description.

    :param pdf_id: The ID of the PDF to parse the skills from.
    :return: The parsed skills.
    """
    aclient = instructor.apatch(AsyncOpenAI(api_key=settings.openai_api_key))

    formatted_user_prompt = user_prompt_job_description.format(
        job_description=job_description.description,
    )

    messages = [
        {"role": "system", "content": system_prompt_job_description},
        {"role": "user", "content": formatted_user_prompt},
    ]

    logging.info(f"Sending messages to OpenAI: {messages}")
    response = await aclient.chat.completions.create(
        model=settings.gpt4_model_name,
        messages=messages,
        response_model=JobDescriptionExtract,  # instructor injection
        response_format=settings.response_format,
        seed=settings.seed,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )

    if response:
        return response
    else:
        raise ValueError("No content received from OpenAI response")


async def parse_skills_cv(pdf_id: int) -> str:
    """
    Evaluate the CV.

    :param pdf_id: The ID of the PDF to evaluate.
    """
    client = OpenAI(api_key=settings.openai_api_key)

    # formatted_user_prompt = user_prompt_job_description.format(
    #     job_description=job_description.description
    # )

    messages = [
        {"role": "system", "content": system_prompt_cv},
        {"role": "user", "content": user_prompt_cv},
    ]

    response = client.chat.completions.create(
        model=settings.gpt4_model_name,
        messages=messages,
        response_format=settings.response_format,
        seed=settings.seed,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )

    return response.choices[0].message.content
