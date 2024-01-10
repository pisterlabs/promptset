import sys
import asyncio

import openai

async def search(query):
    completion = await openai.ChatCompletion.acreate(
        model="ft:gpt-3.5-turbo-0613:personal::8MPbjnyY",
        messages=[
            {"role": "system", "content": "Act as a scryfall api bot that accepts a user query and translates it into a search URL.  Output only the url."},
            {"role": "user", "content": query}
        ],
        temperature=0.0,
    )

    return completion.choices[0].message


def main():
    print(asyncio.run(search(sys.argv[1])))

if __name__ == "__main__":
    main()