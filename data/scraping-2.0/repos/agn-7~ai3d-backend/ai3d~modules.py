import traceback
import g4f
import google.generativeai as genai

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai._types import NOT_GIVEN

from .config import settings
from .database import AsyncSession
from . import OPENAI_MODELS, GPT_TURBO_MODELS, crud, schemas

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
genai.configure(api_key=settings.GEMINI_API_KEY)

g4f.debug.logging = True
g4f.check_version = False


async def openai_response(
    model: schemas.ChatModel, messages: list[schemas.MessageCreate]
) -> ChatCompletion:
    if model in GPT_TURBO_MODELS:
        response_format = {"type": "json_object"}
    else:
        response_format = NOT_GIVEN

    response = await client.chat.completions.create(
        model=model, messages=messages, response_format=response_format
    )

    return response


async def free_response(
    model: schemas.ChatModel, messages: list[schemas.MessageCreate]
) -> str:
    response = await g4f.ChatCompletion.create_async(
        model=g4f.ModelUtils.convert[model], messages=messages
    )
    return response


def gemini_response(
    content: str, interaction: schemas.Interaction
) -> str:  # TODO: needs tracking of messages
    model = genai.GenerativeModel("gemini-pro")
    prompt = interaction.settings.prompt + "\n\n" + content
    response = model.generate_content(prompt)

    return response.text


async def get_recent_messages(
    db: AsyncSession, user_content: str, interaction: schemas.Interaction
) -> list[schemas.MessageCreate]:
    message_objects = await crud.get_messages(
        db=db, interaction_id=str(interaction.id), page=1, per_page=5
    )  # TODO: you should also consider the user

    messages = []
    messages.append({"role": "system", "content": interaction.settings.prompt})
    for message in message_objects:
        messages.append(schemas.MessageCreate.model_validate(message).model_dump())
    messages.append({"role": "user", "content": user_content})

    return messages


async def generate_ai_response(
    db: AsyncSession, content: str, interaction: schemas.Interaction
) -> str:
    try:
        messages = await get_recent_messages(db, content, interaction)

        if interaction.settings.model in OPENAI_MODELS and settings.OPENAI_API_KEY:
            print("OpenAI ...")
            response = await openai_response(
                model=interaction.settings.model, messages=messages
            )
            return response.choices[0].message.content
        elif interaction.settings.model == "gemini-pro" and settings.GEMINI_API_KEY:
            print("Gemini ...")
            response = gemini_response(content, interaction)  # TODO: use from messages
            return response
        else:
            print("G4F ...")
            response = await free_response(
                model=interaction.settings.model, messages=messages
            )
            return response
    except Exception:
        traceback.print_exc()
        return "Sorry, an error has been occurred."
