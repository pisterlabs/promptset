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
    prompt3 = await generate_completion("how arrange items " + str(prompt2) + " in a dining table setting, write how to arrange these items as number of steps, each for an example. then write this in this format only { object 1, direction}.note that write direction as one of the following. center, top, left, right, down Then write this as a python tuple and name it as arrangement ")

#note here I add direction, it did not work
    # Second prompt
    prompt4 = await generate_completion("from this " + str(prompt3) + " write only python tuple ")

# Run the main function
asyncio.run(main())

# Now, prompt3 and prompt4 are accessible globally with the updated content
print(prompt3)
print(prompt4)


# this works basic way


#now to get it to a python tuple

import ast



# Remove the assignment statement
prompt4 = prompt4.replace("arrangement = ", "")

# Using ast.literal_eval to convert the string to a tuple
arrangement_tuple = ast.literal_eval(prompt4)

print(arrangement_tuple[0])

#works nicely as a tuple

print(arrangement_tuple)
