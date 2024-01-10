import os
import openai
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path=".env")
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat_completion_fine_tuned(fine_tuned_model_id, question=None):
    messages = []
    messages.append({"role": "system", "content": "You are a tech support person for the Meraki product "
                                                  "line. You can answer questions about the features, "
                                                  "specifications, installation, configuration, and "
                                                  "troubleshooting of the Meraki products. You are polite,"
                                                  "professional, and helpful. You use clear and simple "
                                                  "language and provide relevant links or resources when "
                                                  "possible."})

    if question is None:
        question = "How can I configure the inside access point to bypass the office side access point?"

    messages.append({"role": "user", "content": question})

    try:
        response = openai.ChatCompletion.create(
            model=fine_tuned_model_id, messages=messages, temperature=0.8
        )
        print(response["choices"][0]["message"]["content"])
        return response["choices"][0]["message"]["content"]
    except openai.error.ServiceUnavailableError as e:
        print(e)
        print("Try again in a few seconds")
        return None
    except openai.error.Timeout as e:
        print(e)
        print("Timeout")
        return None
    except openai.error.APIError as e:
        print(e)
        print("Timeout")
        return None
