import logging

import httpx
import openai
from databases import Database
from openai import AsyncOpenAI

from social.config import config
from social.database import post_table

logger = logging.getLogger(__name__)


class APIResponseException(Exception):
    pass


async def send_simple_email(to: str, subject: str, body: str):
    logger.debug(f"Sending email to '{to[:3]}' with subject '{subject[:10]}'")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"https://api.mailgun.net/v3/{config.MAILGUN_DOMAIN}/messages",
                auth=("api", config.MAILGUN_API_KEY),
                data={
                    "from": f"David <mailgun@{config.MAILGUN_DOMAIN}>",
                    "to": [to],
                    "subject": subject,
                    "text": body,
                },
            )
            response.raise_for_status()
            logger.debug(response.content)
            logger.debug(f"Email sent to {to[:3]} with subject {subject}")
            return response

        except httpx.HTTPStatusError as e:
            logger.error(f"Error sending email: {e}")
            raise APIResponseException(
                f"API request with status code {e.response.status_code} failed"
            ) from e


async def send_user_registration_email(to: str, confirmation_url: str):
    subject = "Please confirm your email"
    body = f"""
    Hi there,
    You have successfully registered signed up!
    Please confirm your email by clicking on the link below:
    {confirmation_url}
    """
    await send_simple_email(to, subject, body)
    logger.debug(f"Confirmation email sent to {to[:3]} with subject {subject}")


async def _generate_image_api(prompt: str):
    logging.debug(f"Generating image from prompt: {prompt[:30]}")
    openai_client = AsyncOpenAI(
        api_key=config.OPENAI_API_KEY,
        max_retries=1,
        timeout=60,
    )
    try:
        response = await openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size=config.OPENAI_IMAGE_SIZE,
        )
        logger.debug(response)
        output = response.model_dump(exclude_unset=True)
        logger.debug(f"The response form openai is {output}")
        return output

    except openai.APIConnectionError as e:
        # Handle connection error, e.g. check network or log
        print(f"OpenAI API request failed to connect: {e}")
        pass
    except openai.BadRequestError as e:
        # Handle invalid request error, e.g. validate parameters or log
        print(f"OpenAI API request was invalid: {e}")
        pass
    except openai.AuthenticationError as e:
        # Handle authentication error, e.g. check credentials or log
        print(f"OpenAI API request was not authorized: {e}")
        pass
    except openai.PermissionDeniedError as e:
        # Handle permission error, e.g. check scope or log
        print(f"OpenAI API request was not permitted: {e}")
        pass
    except openai.RateLimitError as e:
        # Handle rate limit error, e.g. wait or log
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass
    except openai.APIStatusError as e:
        # Handle rate limit error, e.g. wait or log
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass
    except openai.APIError as e:
        # Handle API error, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass


async def generate_image_and_add_to_post(
    email: str,
    post_id: int,
    post_url: str,
    database: Database,
    prompt: str,
):
    try:
        response = await _generate_image_api(prompt)
    except APIResponseException as e:
        logger.error(f"Error generating image: {e}")
        to = email
        subject = "Error generating image"
        body = (
            "Hi there,\nUnfortunately there was an error "
            "generating an image for your post"
        )
        return await send_simple_email(to, subject, body)

    logger.debug("Connection to database to update image_url")
    image_url = response["data"][0]["url"]
    query = (
        post_table.update()
        .where(post_table.c.id == post_id)
        .values(image_url=image_url)
    )
    logger.debug(query)
    await database.execute(query)
    logger.debug(f"Database background task for {post_id} closed")
    to = email
    subject = "Image generated for your post!"
    body = (
        f"Hi there,\nYour image has been generated"
        f" for your post and is available at {post_url}.\n"
    )
    await send_simple_email(to, subject, body)
    return response
