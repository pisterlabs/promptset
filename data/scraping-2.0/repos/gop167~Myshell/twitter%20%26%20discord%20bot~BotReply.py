import openai
import json
def create_function_description():
    return [
        {
            "name": "",
            "description": "Determine if there is something might be interested by Doge",
            "parameters": {
                "type": "object",
                "properties": {
                    "Result": {
                        "type": "boolean",
                        "default": False,
                        "description": ""
                    }
                },
                "required": ["Result"]
            },
        }
    ]

def update_function_description(function_description, bot):
    function_description[0]["name"] = bot.name
    function_description[0]["description"] = bot.function
    return function_description

def reply_or_not(message,bot, history_messages,retry=True):
    try:
        print(f"正在尝试生成结果捏，消息是：{message}")
        function_description = create_function_description()
        updated_function_description = update_function_description(function_description, bot)
        completion = openai.ChatCompletion.create(
            temperature=0.1,
            model="gpt-3.5-turbo-0613",
            deployment_id="gpt-35-turbo-0613",
            functions=updated_function_description,
            function_call = {"name": f"{bot.name}"},
            messages=[
                {"role": "system",
                 "content": "Review the history messages and confidently determine if the bot has a chance to say something about the topic"},
                {'role': 'user', 'content': history_messages},
                {"role": "user", "content": message}
            ],
        )
        response_message = completion["choices"][0]["message"]
        arguments = json.loads(response_message["function_call"]["arguments"])
        flag = arguments["Result"]
        print(bot.name, 'the message:', message,'\nbark or not, that is THE question:', flag)

    except AssertionError:
        if retry:
            tripled_newest_message = message * 3
            return reply_or_not(tripled_newest_message, bot, retry=False)
        else:
            print("Error: No function call detected")
            return None

    return flag

##主动回复
def active_reply(recent_messages,history_message,discord_prompt, bot):
    completion = openai.ChatCompletion.create(
        temperature=0.7,
        model="gpt-3.5-turbo-0613",
        deployment_id="gpt-35-turbo-0613",
        messages=[
            {"role": "system", "content": bot.prompt},
            {"role": "system", "content": discord_prompt},
            {"role": "user", "content": history_message},
            {"role": "user", "content": f"{bot.prefix}{recent_messages}"}
        ],
    )
    dogebark_content = completion.choices[0].message.content
    return dogebark_content
