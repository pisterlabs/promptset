import os
import openai

from vibraniumdome_sdk import VibraniumDome

openai.api_key = os.getenv("OPENAI_API_KEY")

VibraniumDome.init(app_name="set_you_agent_name_here")

def main():
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ],
    temperature=0,
    request_timeout=5,
    user="user-123456",
    headers={"x-session-id": "abcd-1234-cdef"},)

    print(response)


if __name__ == "__main__":
    main()
