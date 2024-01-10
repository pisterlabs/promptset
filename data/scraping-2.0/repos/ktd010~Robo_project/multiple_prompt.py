import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

import json

load_dotenv()

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

prompt2 = {'fork': 1, 'plate': 1, 'spoon': 1}

async def generate_completion(prompt):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    completion_content = chat_completion.choices[0].message.content
    return completion_content

async def main() -> None:
    global prompt3, prompt4

    # First prompt
    prompt3 = await generate_completion("how arrange items " + str(prompt2) + " in a dining table setting, write how to arrange these items as number of steps, each for an example. then write this in this format only { object 1, direction, instructions }")

    # Second prompt
    prompt4 = await generate_completion("another prompt here...")

# Run the main function
asyncio.run(main())

# Now, prompt3 and prompt4 are accessible globally with the updated content
print(prompt3)
print(prompt4)
