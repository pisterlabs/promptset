import json
import openai

import json

openai.api_key = "xxxxxxxxxx"
openai.api_base = "https://xxxxxx-975.openai.azure.com/"
openai.api_type = "azure"
deployment_name = "turbo0613"  # T
openai.api_version = "2023-07-01-preview"


def init_meta_prompt() -> any:
    # print("init")
    # read all lines from a text file
    with open("metaprompt-1.txt", "r") as file:
        data = file.read().replace("\n", "")
    # print(data)
    chat_history = [{"role": "system", "content": data}]
    return chat_history


def send_message_llm(message: str, chat_history: any) -> str:
    chat_history.append({"role": "user", "content": message})
    # print(chat_history)
    response_message = openai.ChatCompletion.create(
        engine=deployment_name, messages=chat_history, temperature=0
    )

    llm_response = response_message["choices"][0]["message"]["content"]

    # use substring to extract json data based on start '{' character and end '}' character
    try:
        json_response = llm_response[
            llm_response.find("{") : llm_response.rfind("}") + 1
        ]
        if json_response == "":
            print(llm_response)
            chat_history.append({"role": "assistant", "content": llm_response})
            return llm_response
        else:
            # print('the json in the response is',json_response)
            resp_object = json.loads(json_response)
            age = resp_object["age"]
            annual_income = resp_object["annual_income"]
            current_savings = resp_object["current_savings"]
            response = run_business_rules_finance_advisor(
                age, annual_income, current_savings
            )
            chat_history.append({"role": "assistant", "content": response})
    except:
        return ""
        pass


def run_assistant():
    chat_history = init_meta_prompt()
    while True:
        message = input(">> ")
        if message == "quit":
            break
        else:
            send_message_llm(message, chat_history)


def run_business_rules_finance_advisor(age, annual_income, current_savings):
    print(
        "Running business rules engine with these input.. \n age: ",
        age,
        "annual_income: ",
        annual_income,
        "current_savings: ",
        current_savings,
    )
    try:
        with open("metaprompt-2.txt", "r") as file:
            data = file.read().replace("\n", "")
        user_input = (
            "age: "
            + str(age)
            + "\n annual_income: "
            + str(annual_income)
            + "\n current_savings: "
            + str(current_savings)
        )
        # print(data)
        chat_history = [
            {"role": "system", "content": data},
            {"role": "user", "content": user_input},
        ]
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=chat_history,
            temperature=0,
        )
        llm_response = response["choices"][0]["message"]["content"]
        print(llm_response)
        return llm_response
    except Exception as e:
        print("Error in running business rules engine", e)
        return "Error in running business rules engine"


if __name__ == "__main__":
    run_assistant()
