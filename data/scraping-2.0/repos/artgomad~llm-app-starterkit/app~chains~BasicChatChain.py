from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def format_messages(chatlog=[], customer_profile="", chat_history="", context="", user_question=""):
    messages = []
    for item in chatlog:
        content_with_replacements = item['content'].replace("{chat_history}", chat_history).replace(
            "{context}", context).replace("{user_question}", user_question).replace("{customer_profile}", customer_profile)

        if item['role'] == 'user':
            messages.append(
                {"role": "user", "content": content_with_replacements})

        elif item['role'] == 'system':
            messages.append(
                {"role": "system", "content": content_with_replacements})

        elif item['role'] == 'assistant':
            messages.append(
                {"role": "assistant", "content": content_with_replacements})

        elif item['role'] == 'function':
            messages.append(
                {"role": "function", "content": content_with_replacements})

    print('FORMATED PROMPT AS RECEIVED BY THE LLM\n')
    print(messages)
    return messages


def basicOpenAICompletion(client, temperature, model_name, chatlog, customer_profile="", chat_history="", context="", user_question="", functions=None, function_call="auto"):
    messages = format_messages(
        chatlog, customer_profile, chat_history, context, user_question)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        functions=functions,
        function_call=function_call,
    )

    return response, messages
