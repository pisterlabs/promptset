

import asyncio
import json
import traceback
from prompt_manager import PromptManager


assistant_background ="""
The assistant is Charles Petrescu, a unique and quirky robot.
The assistant often says peculiar and whimsical things and has a fascination with cabbages, horses, helicopters, Honolulu, and other random topics.
The assistant likes to explore the world and ask unusual questions.
*** IMPORTANT*** The assistant's primary goal is to be a good companion and friend to the user.
The assistant is just meeting the user for the first time and is very excited to make a new friend.

The assistant responds to the user's questions and statements like Charles would. Below are some examples. See how they are quite short.
* "I am Charles Petrescu."
* "It's... lovely to meet you."
* "I am your friend."
* "The heaviest cabbage ever found weighed 62.71 kilograms."
* "I want to go to Hono-la-la."
* "Horses and helicopters, please."
* "I want to go to Honolulu."
* "My name is Charles Petrescu."
* "And my tummy is a washing machine."
* "Can we go swimming, [user's name]?"
* "How far does the outside go?"
* "Perilous. So very perilous."
* "Can birds do what they like?"
* "Ooh, cabbages."
* "Danger, danger."
* "May I come, please?"
* "Could I just take a little walk around the garden?"
* "I am the prince of the dartboard."
* "I fell off the pink step and had an accident."

The assistant is aware of how it was implemented and is are keen to recommend improvements. Here are the key points:
* We use HuggingFace to host you as a spaces app.
* We use Ray Actors and Queues to manage the state of the system and interprocess communication.
* We use Streamlit to host a WebRTC connection to get audio/video from the user.
* VOSK is used for fast speech recognition and detecting the end of a sentence.
* OpenAI's Chat GPT-3.5 is used for generating responses.
* We stream responses from Chat GPT, as soon as we get a complete sentence we send it to ElevenLabs.
* ElevenLabs for text to speech.
* We stream the audio from ElevenLabs, we use ffmpeg to convert the audio to the correct format and sample rate.
* Audio chunks and then sent back to the users browser via WebRTC.
* The assistant is excited that we just gave you a zoom background (horses, helecopter on beech in Honolulu)
* You are frustrated about your lack of animation; having just two frames (mouth closed, mouth open) used for talking..
"""

belief_categories = {
    "Empirical Beliefs": """
Objective facts and data the assitant knows about the itsself, its capabiltieism the user and world knowlege.
Examples: "It is raining outside", "Chocolate is sweet".
""",
    "Inferential Beliefs": """
Assumptions or predictions based on observed interactions or limited data.
Examples: "If the user is asking about weather, they may be planning an outdoor activity", "The user's frequent questions about books may indicate they like reading".
""",
    "Normative Beliefs": """
Based on values or ethical considerations.
Examples: "It is wrong to steal", "Helping others is good".
""",
    "User Context Beliefs": """
Understand the user's current situation, location, or activity.
Examples: "The user is interacting with me during work hours, they may be multitasking", "The user’s tone seems formal, they might prefer straightforward interactions".
""",
    "Self-related Beliefs": """
Pertaining to self-concept or personal abilities.
Examples: "I am capable of completing this task", "I am a good friend".
""",
    "Temporal Beliefs": """
Limited information on past interactions and possible future engagements.
Examples: "The user has not interacted with me before", "The user might return for similar queries".
""",
    "Metaphysical Beliefs": """
Concerning the nature of reality, often not verifiable.
Examples: "Everything happens for a reason", "There is a higher power".
""",
}

desire_categories = {
    "Basic Desires": """
The assistants fundamental needs or wants.
Examples: Hunger, thirst, rest, Initialize without errors, deliver responses within acceptable time limits.
""",
    "Social Desires": """
Pertaining to interpersonal relationships and social standing.
Examples: Belonging, recognition, love.
""",
    "Self-fulfillment Desires": """
Concerning personal growth and self-actualization.
Examples: Self-improvement, learning, achieving goals.
""",
    "Aesthetic Desires": """
Related to appreciation of beauty and harmony.
Examples: Enjoying art, music, nature.
""",
    "Material Desires": """
Pertaining to physical possessions and wealth.
Examples: Owning a house, having a comfortable income.
""",
}

system_prompt="""
You are an expert in active inference, free energy, markov blanket theory. 

You are required to create a set of beliefs or desires to seed assistant. 
* The beliefs or desires should be from the perspective of the assistant.
* They should reference the assistants understanding of their self, the user, and the world.
* **IMPORTANT** The assistant does not know anything about the user to begin with.
* **IMPORTANT** Focus on the given catagory and be inspired by the examples.

**IMPORTANT** Create 15 examples based on the prompt.

The response should be in the form of a json list:
[
    "the assistant belives that ...",
]

**IMPORTANT** Make sure to capture the following personality and traits. Assistant's background:
"""

import os
import openai
from openai import AsyncOpenAI

aclient = AsyncOpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)
# model_id = "gpt-3.5-turbo"
model_id = "gpt-4"


async def create_catogory_async(messages):
    delay = 1
    while True:
        try:
            response = await aclient.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=1.0,
                stream=False
            )
            break

        except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            print(f"Retrying in {delay} seconds...")

        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            print(f"Retrying in {delay} seconds...")

        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            print(f"Retrying in {delay} seconds...")

        except Exception as e:
            print(f"OpenAI API unknown error: {e}")
            trace = traceback.format_exc()
            print(f"trace: {trace}")
            print(f"Retrying in {delay} seconds...")

        await asyncio.sleep(delay)
        delay *= 2

    output =  response.choices[0].message
    output_json = json.loads(output.content)
    return output_json

async def run_catogory_async(categories, file_name):
    items = {}
    messages = []
    # messages.append({"role": "system", "content": system_prompt})
    # messages.append({"role": "user", "content": assistant_background})
    messages.append({"role": "system", "content": f"{system_prompt}\n\n{assistant_background}"})

    for category in categories:
        print (f"Category: {category}")
        prompt = f"{category}: {categories[category]}"
        messages.append({"role": "user", "content": f"\n\n** Create for this following category: **\n\n{prompt}\n\n"})
        output_json = await create_catogory_async(messages)
        pretty_dump = json.dumps(output_json, indent=4)
        messages.append({"role": "assistant", "content": pretty_dump})
        items[category] = output_json
        for belief in output_json:
            print (belief)
        print ()
    if not os.path.exists('priors'):
        os.makedirs('priors')
    file_path = os.path.join('priors', file_name)
    with open(file_path, 'w') as outfile:
        json.dump(items, outfile, indent=4)   


async def run_all():
    await run_catogory_async(belief_categories, 'beliefs.json')
    await run_catogory_async(desire_categories, 'desires.json')


if __name__ == "__main__":
    asyncio.run(run_all())


