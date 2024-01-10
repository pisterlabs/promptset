try:
    import secret_values
except ImportError:
    pass
import os
import requests
from openai import OpenAI

# gpt =================
gpt_key = os.environ.get("GPT_KEY", default=None)
if not gpt_key:
    gpt_key = secret_values.GPT_KEY

gpt_client = OpenAI(api_key=gpt_key)

# search ========================
subscription_key = os.environ.get("SEARCH_KEY", default=None)
if not subscription_key:
    subscription_key = secret_values.SEARCH_KEY
assert subscription_key
search_url = "https://api.bing.microsoft.com/v7.0/search"

assistant = gpt_client.beta.assistants.create(
    name="fact_checker",
    instructions="your task is to read the input and determine it's misleading. If the message is misleading, explain. your task is to read text and tell me if it's misleading, if it is, tell me why. If you don't have the real time information, your reply will only be the number: 1, nothing else should be returned",
    model="gpt-4-1106-preview",
)


def if_misinfo(message):
    result = ""
    thread = gpt_client.beta.threads.create(
        messages=[{"role":"user", "content":message}]
    )
    run = gpt_client.beta.threads.runs.create(
        thread_id= thread.id,
        assistant_id=assistant.id,
    )

    # wait for the assistant to respond
    while run.status != "completed":
        run = gpt_client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    
    message_lst = gpt_client.beta.threads.messages.list(thread_id=thread.id)
    last_msg = message_lst.data[0].content[0].text.value
    print(last_msg)
    if last_msg != "1":
        return last_msg

    # knowledge from self

    info_on_web = search_result(message)
    info_on_web = str(info_on_web)
    prompt = f"given this search phrase: {message}\nand this results:{info_on_web},\n determine if this message:{message} is misleading, keep the response simple."
    stream = gpt_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            word = chunk.choices[0].delta.content
            result += word
    return result


# def gen_search_phrase(message):
#     new_message = "is " + message + " correct?"
#     new_message = "correct this sentence:" + new_message
#     search_phrase = ""
#     stream = gpt_client.chat.completions.create(
#         model="gpt-4-1106-preview",
#         messages=[{"role": "user", "content": new_message}],
#         stream=True,
#     )
#     for chunk in stream:
#         if chunk.choices[0].delta.content is not None:
#             word = chunk.choices[0].delta.content
#             search_phrase += word
#     search_phrase = search_phrase.replace('"', "")
#     search_phrase = search_phrase.replace("'", "")

#     return search_phrase


def search_result(message):
    print(message)
    # search_term = gen_search_phrase(message)
    # print(search_term)
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": message, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    print(search_results)
    result_lst = []
    i = 0
    for v in search_results["webPages"]["value"]:
        if i == 5:
            break
        i += 1
        result_lst.append((v["name"], v["snippet"]))
    return result_lst
