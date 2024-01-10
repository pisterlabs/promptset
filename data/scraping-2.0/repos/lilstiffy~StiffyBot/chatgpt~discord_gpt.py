from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

is_initialised = False
openAiClient = None
MODEL = "gpt-3.5-turbo"

try:
    load_dotenv()
    is_initialised = True
    openAiClient = AsyncOpenAI(api_key=os.getenv('OPEN_AI_TOKEN'))
except Exception as init_error:
    print(f"Could not initialise OpenAI client: {init_error}")


async def chat_with_gpt(input_text):
    try:
        response = await openAiClient.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": input_text}
            ]
        )

        return response.choices[0].message.content

    except Exception as gpt_error:
        if is_initialised:
            return f"An error occurred: {str(gpt_error)}"
        else:
            return f"Missing OpenAI key"
