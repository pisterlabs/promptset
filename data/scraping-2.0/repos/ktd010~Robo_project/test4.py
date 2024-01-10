
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


prompt2={'fork': 1, 'plate': 1, 'spoon': 1}



# Declare prompt2 as a global variable
prompt3 = {}

async def main() -> None:
    global prompt3

    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "how arrange items " + str(prompt2) + "in a diining table setting,   write how to arrange these items as number os steps , each for item for an examle. then write this  in to this format only  { object 1, direction, instructions }  " ,
            }
        ],
        model="gpt-3.5-turbo",
    )
    
    completion_content = chat_completion.choices[0].message.content
    #print(completion_content)
    prompt3 = completion_content
    #print(prompt3)

# Run the main function
asyncio.run(main())

# Now, prompt2 is accessible globally with the updated content_dict
print(prompt3)

