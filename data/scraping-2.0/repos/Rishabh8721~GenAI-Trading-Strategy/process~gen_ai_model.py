from openai import OpenAI

from process import prompt_generator
from util import secrets_util

client = OpenAI(
    api_key=secrets_util.get_openai_api_key(),
)

messages = []


def chat(message):
    if not messages:
        return get_initial_analysis(message)
    elif "@@clear" in message:
        clear_message_thread()
        return "Model has been resetted"
    elif "@@" in message:
        company_info = message.split("@@")
        return get_initial_analysis(company_info[0], company_info[1])
    else:
        return follow_up_question(message)


def _get_ticker_symbol(company_name):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": "Give just ticker symbol and nothing else for " + company_name}],
        stream=False,
    )

    ticker_symbol = response.choices[0].message.content
    print(ticker_symbol)
    return ticker_symbol


def get_initial_analysis(company_name, symbol=None, live=True):
    clear_message_thread()
    messages.append({"role": "system", "content": "You are a Financial Analyst. You have to analyze financial data of "
                                                  "a company and suggest investment strategy for that company. You "
                                                  "only have to answer in context to the specified company."
                                                  "Response must not include 'As per analysis', "
                                                  "'as per given data' etc."})
    if symbol is None:
        symbol = _get_ticker_symbol(company_name)

    messages.append({"role": "user", "content": prompt_generator.generate_prompt(company_name, symbol, live)})

    return execute_and_respond()


def follow_up_question(question):
    messages.append({"role": "user", "content": "(Only Consider Previous Data) " + question})
    return execute_and_respond()


def execute_and_respond():
    print(messages)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        stream=False,
    )

    response_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": response_msg})
    return response_msg


def clear_message_thread():
    messages.clear()
