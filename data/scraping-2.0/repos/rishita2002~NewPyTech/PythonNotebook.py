# Asyncio, Python Type Hints, Pydantic, OpenAI API (mocked) and Docker.


# Section 1: Asyncio and Python Type Hints

# Asyncio enables asynchronous programming in Python, improving concurrency
# Type hints enhance code readability and catch potential errors.

import asyncio

async def intro(name : str, age : int):
    print("Hello World!")
    await asyncio.sleep(1)
    print(f"This is {name},  I am {age} now.")
    await asyncio.sleep(4)
    print("Byee, signing off!")

async def brief(college : str, degree : str, branch : str, semester : int, year : int):
    await asyncio.sleep(2)
    print(f"I am a student of {college}")
    await asyncio.sleep(1)
    print(f"I am pursuing {degree} in {branch}")
    await asyncio.sleep(1)
    print(f"Currently I am in {year}th year, {semester}th semester.")


async def main():
    task1 = intro("Rishita Ghosh", 21)
    task2 = brief("GEC-R", "BTech", "CSE", 7, 4)
    await asyncio.gather(task1, task2) # gather is a ayncio method to run task1 and task2 concurrently

asyncio.run(main()) # event loop to run main
# always needs to be called from the main block and not already running event loop enablling parallel execution

#-------------------------------------------------------------------------------------------------------------------

# Section 2: Pydantic
# Pydantic simplifies data validation and parsing in Python

from pydantic import BaseModel

class Person(BaseModel):

    # attributes
    Name: str 
    Age: int
    # something similar to python type hinting

# Example usage:
person_data = {"Name": "Rishita", "Age": 21}
validated_person = Person(**person_data)

''' **person_data - unpacks the dictionary, passing its key-value pairs as keyword arguments to the Person constructor.

Person(**person_data) - creates an instance of the Person model with its fields populated based on the values in the person_data dictionary.'''

print(validated_person)

#--------------------------------------------------------------------------------------------------------------------------------

# Section 3: OpenAi API (Dummy)

"""import openai

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

"""

def mock_translate_text(source_text: str, source_language: str, target_language: str) -> str:

    """
    prompt = f"Translate the following text from {source_language} to {target_language}:\n{source_text}"   ----> prompt string spcifying the task
    
    response = openai.Completion.create(         ----------> method of openai gpt 3 api that generates 
                                                                                    response from the language model by passing parameters


        engine="text-davinci-003",  ----> specifying the engine to use with openai gpt-3 language model

        prompt=prompt,  -----> passing the prompt as the prompt we set above

        max_tokens=50 ---> limiting length of generated text
    )

    return response.choices[0].text.strip()

    """

    mock_response = {
        "choices": [{"text": f"Translated: {source_text} from {source_language} to {target_language}."}]
    }
    
    return mock_response["choices"][0]["text"]

# Example usage:
translated_text = mock_translate_text("Hello, world!", "English", "French")
print(f"Translated Text: {translated_text}")
