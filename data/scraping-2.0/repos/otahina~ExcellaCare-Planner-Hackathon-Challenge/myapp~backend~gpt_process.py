import openai
import os
from dotenv import load_dotenv

from summarize_prompt import summarize_prompt

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# User message is passed
def build_conversation(user_message):
    return [
        {
            "role": "system",
            "content": "You are a schedule organizer for radiotherapy treatments to use machines efficiently."
                       "The return format must be: Date: {date} /n Machine: {machine name} /n Available time: {time}"
        },
        {
            "role": "user",
            "content": user_message
        }
    ]


# Generate response
def generate_assistant_message(conversation):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    return response['choices'][0]['message']['content']


def generate_response(user_message):
    conversation = build_conversation(user_message)
    try:
        assistant_message = generate_assistant_message(conversation)
    except openai.error.RateLimitError:
        return "Rate limit exceeded. Sleeping for a bit..."

    except Exception as e:
        return f"An error occurred: {e}"

    return assistant_message


# Test
if __name__ == "__main__":
    user_message = summarize_prompt("normal")
    response = generate_response(user_message)
    print(response)
