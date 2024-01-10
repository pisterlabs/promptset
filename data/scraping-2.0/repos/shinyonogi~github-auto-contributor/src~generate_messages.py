import logging
import openai


message_to_gpt = f"""
Give me an interesting random fact in one sentence.
"""
message = [{"role": "user", "content": message_to_gpt}]


def gpt_commit_message_generation():
    try:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-0301",
            messages = message
        )
        return response.choices[0].message.content
    except openai.error.OpenAIError as e:
        logging.error(f"Error with OpenAI API: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occured: {e}")
    return "Default Commit Message"
