import re
import pprint
import requests
from openai_utils import get_code_from_open_ai
from html_cleaner import get_cleaned_html


def send_message(message):
    api_url = "http://127.0.0.1:8000/"
    json_payload = {"code": message}
    response = requests.post(api_url, json=json_payload)
    # Check the response status code
    if response.status_code == 200:
        print("Request successful. Response:")
        response_data = response.json()
        print(response_data)
        return response_data
    else:
        print(f"Request failed with status code: {response.status_code}")


def main():
    history_messages = []
    while True:
        action = input("Enter a web action or exit to quit: ")
        if action.lower() == "exit":
            break
        is_error = input("Is this an error? (y/n): ")
        is_error = is_error.lower() == "y"
        history_messages, message = get_code_from_open_ai(
            action, history_messages, is_error
        )
        pprint.pprint(history_messages)
        response = send_message(message)


if __name__ == "__main__":
    main()
