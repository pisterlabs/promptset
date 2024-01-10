import os
import openai

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

history = []


def mount_messages(content, role="user"):
    history.append({"role": role, "content": content})
    return history


def extract_openia_response(response):
    returned_response = response['choices'][0]['message']['content']
    return returned_response


def get_ia_response(new_message):
    messages = mount_messages(new_message)

    try:
        response = openai.ChatCompletion.create(
            temperature=0.7,
            messages=messages,
            model="gpt-3.5-turbo",
        )

        extracted_response = extract_openia_response(response)

        mount_messages(extracted_response, role="assistant")

        return extracted_response
    except Exception as e:
        print(e)
