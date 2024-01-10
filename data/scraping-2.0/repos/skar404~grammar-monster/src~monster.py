import json

import openai

from settings import settings


async def check_grammar(message: str) -> str | None:
    openai.api_key = settings.openai_token

    functions = [
        {
            "name": "check_grammar",
            "description": "It is called for each user message. "
                           "A function that praises the user in case there are no grammatical "
                           "errors or notifies them about the errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "have_mistakes": {
                        "type": "boolean",
                        "description": "Boolean flag, return True - have mistakes else False",
                    },
                    "description": {
                        "type": "string",
                        "description": "A description of what the user did wrong and suggestions on how to improve."
                    },
                },
                "required": ["have_mistakes", "description"],
            },
        }
    ]

    messages = [{"role": "user", "content": message}]

    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )

    response_message = response["choices"][0]["message"]
    function_call = response_message.get("function_call")
    if not function_call:
        return None

    function_args = json.loads(response_message["function_call"]["arguments"])
    if not function_args.get('have_mistakes') or not function_args.get('description'):
        return None

    return function_args.get('description')


if __name__ == '__main__':
    # debug code
    import asyncio


    async def main():
        r = await asyncio.gather(*[check_grammar(m) for m in [
            "I has been working on this project for a long time.",
            "He don't like to eat vegetables.",
            "Their going to the party tonight.",
            "She don't have any money to buy a new car.",
            "Can you borrow me your pen, please?",
            "I seen that movie last night and it was great.",
            "The cat chased it's tail around in circles.",
            "He has went to the store to buy some groceries."
        ]])
        for m in r:
            print(m)
            assert m


    asyncio.run(main())
