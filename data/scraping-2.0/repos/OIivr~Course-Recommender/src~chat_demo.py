import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_type = "azure"
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = "https://tu-openai-api-management.azure-api.net/OLTATKULL"
openai.api_version = "2023-07-01-preview"


def chat(system, user_assistant):
    assert isinstance(system, str), "`system` should be a string"
    assert isinstance(
        user_assistant, list), "`user_assistant` should be a list"
    system_msg = [{"role": "system", "content": system}]
    user_assistant_msgs = [
        {"role": "assistant", "content": user_assistant[i]} if i % 2 else {
            "role": "user", "content": user_assistant[i]}
        for i in range(len(user_assistant))]

    msgs = system_msg + user_assistant_msgs
    try:
        response = openai.ChatCompletion.create(
            deployment_id="IDS2023_PIKANI_GPT35",
            model="gpt-3.5-turbo",  # specify the model
            temperature=0.0,  # Setting the temperature
            messages=msgs
        )
        status_code = response["choices"][0]["finish_reason"]
        assert status_code == "stop", f"The status code was {status_code}."

        # ----------------LOGS THE TOKENS USED----------------- #
        with open("TOTAL_TOKENS_USED.txt", 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('gpt_tokens_used='):
                total_tokens_used = int(line.split('=')[1])
                break

        total_tokens_used += response['usage']['total_tokens']
        print("----------------------------------------------")
        print("Tokens used this request:", response['usage']['total_tokens'])
        print("Total tokens used:", total_tokens_used)
        print("----------------------------------------------")

        for i, line in enumerate(lines):
            if line.startswith('gpt_tokens_used='):
                lines[i] = f'gpt_tokens_used={total_tokens_used}\n'
                break
        with open("TOTAL_TOKENS_USED.txt", 'w') as f:
            f.writelines(lines)
        # ------------------------------------------------------ #

        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("An error occurred:", e)
        return None


# ---------------------TESTING--------------------- #
result = chat("You are a Course Recommender Assistant in the Univerity of Tartu.",
              ["How many ECTS is “Introduction to data science” courses?"])

print(result)
