import openai

from chats.client_utils import create_client

create_client()


def fix(
    prompt: str,
    instruction: str = "Please clean up the text, fix spelling, make it sound educated.",
):
    response = openai.Edit.create(
        model="text-davinci-edit-001",
        input=prompt,
        instruction=instruction,
    )
    return response
