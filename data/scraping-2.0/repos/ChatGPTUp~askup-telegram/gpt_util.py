import os

import openai
import logging

CHATGPT_ENGINE = os.getenv("CHATGPT_ENGINE", "gpt-3.5-turbo")
BOT_PENCIL_ICON = os.getenv("BOT_PENCIL_ICON", "✍")

UPDATE_CHAR_RATE = 5

openai.api_key = os.environ["OPENAI_API_KEY"]

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def chatgpt_response(messages):
    try:
        response = openai.ChatCompletion.create(
            model=CHATGPT_ENGINE,
            messages=messages
        )
        return response.choices[0].message.content
    except (KeyError, IndexError) as e:
        return "GPT3 Error: " + str(e)
    except Exception as e:
        return "GPT3 Unknown Error: " + str(e)


async def chatgpt_callback_response(messages, call_back_func, call_back_args):
    content = ""

    try:

        response = openai.ChatCompletion.create(
            model=CHATGPT_ENGINE,
            messages=messages,
            stream=True,
        )

        # Stream each message in the response to the user in the same thread
        counter = 0
        for completions in response:
            counter += 1
            if "content" in completions.choices[0].delta:
                content += completions.choices[0].delta.get("content")

            if call_back_func and call_back_args:
                if counter % UPDATE_CHAR_RATE == 0:
                    # Send or update the message,
                    # depending on whether it's the first or subsequent messages
                    call_back_args['text'] = content+BOT_PENCIL_ICON
                    await call_back_func(**call_back_args)

        return content

    except (KeyError, IndexError) as e:
        return "GPT3 Error: " + str(e)
    except Exception as e:
        logger.error("GPT3 Unknown Error: " + str(e))
        if content:
            return content
        return "GPT3 Unknown Error: " + str(e)
