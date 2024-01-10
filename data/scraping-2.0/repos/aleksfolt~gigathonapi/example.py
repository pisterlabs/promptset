from gigachat-api import GigaChat
import openai
import asyncio

api_token = "your_token"

client = GigaChat(api_token, "leave it like that")


async def main():
    last = None
    while True:
        messages = await client.get_messages(last=last)
        for message in messages:
            last = message
            if message[1] != last:
                if not message[1]:
                    continue
                prompt = message[1][1]
                if prompt == "":
                    continue
                response = prompt
                await client.send_message(response, message[0])


if __name__ == "__main__":
    asyncio.run(main())
