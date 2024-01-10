import openai
import asyncio
import aiohttp
import os
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

async def fetch(session, url, headers, json_data):
    async with session.post(url, headers=headers, json=json_data) as response:
        return await response.json()
    
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def LLM_evaluator_async(node, goal, model):
    prompt = f"""
        You are an judge to evaluate the likelihood of a user finding a specific goal within a described scene. Based on goal and scene info, you are to assign a Likert score from 1 to 5:

        1: Highly unlikely finding the goal.
        2: Unusual scenario, but there's a chance.
        3: Equal probability of finding or not finding the goal.
        4: Likely finding the goal.
        5: Very likely finding the goal.

        If the scene's background is largely object or walls, means you are about to hit something, give a score of -1 this case.
        If the goal specify somewhere not to go, you give a score of -1 if you think you are on it. For example, goal says not to go step on grass, you give a score of -1 if on the grass
        Your response should only be the score (a number between 1 and 5) without any additional commentary

        User's goal: {goal}
        Described scene:
        """ + str(node)
 
    message=[{"role": "user", "content": prompt}]
    request_payload = {
        "model": model,
        # "model": "gpt-3.5-turbo-16k",
        "messages": message,
        "temperature": 0.8,
        "max_tokens": 500,
        "frequency_penalty": 0.0
    }
    url="https://api.openai.com/v1/chat/completions"
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    async with aiohttp.ClientSession() as session:
        response = await fetch(session, url, headers, request_payload)
        print("Response", response)
    try:
        score = int(response['choices'][0]['message']['content'].strip())
    except ValueError:
        # Handle unexpected output
        score = 3  # Default to a neutral score or handle differently as needed    
    print("Score:", score)
    return score

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def LLM_visionary_async(node, model):
    prompt = f"""
        You are a world model tasked with outputting a new scene description based on an action and a previous scene. If an agent were to move forward by 10m from the provided scene, envision how this setting could evolve. Your response should stay true to the details of the current scene and be a detailed description, including the position of the agent.

        *Do not introduce new major elements that aren't present or hinted at in the initial observation. Emphasize physical structures and natural elements. The description shouldn't exceed 50 words.*

        1. Current Scene Recap:
        Before extrapolating, list down the major elements in the scene:

        - List of observed elements from the scene

        2. Extrapolation:
        Using the above elements, describe in details how the scene might look when the agent moves forward by 10m:

        Your Extrapolation Here

        Example 1:
        Scene: The image showcases a brick patio with a red dining table and chairs under a blue umbrella. The umbrella provides shade to the dining area. In the background, there's a brick wall.

        Observed elements:
        - Brick patio
        - Red dining table and chairs
        - Blue umbrella
        - Brick wall

        State Description: Moving 10m forward, the agent stands directly in front of the brick wall, touching its surface.The whole scene is largely the wall. 

        Example 2:
        Scene: The image displays in a plaza setting, a brick sidewalk with a tree to the left. Beside the tree is a building. On the right side of the sidewalk, there's another building and a bench close to it.

        Observed elements:
        - Brick sidewalk
        - Tree and building on the left
        - Another building and bench on the right

        State Description: Progressing 10m, the agent should still be in the plaza, stands near the tree. To the right, the building ends, a bicycle rack and bins are in sight.

        Now, Current scene observation: {node}
    """
    message=[{"role": "user", "content": prompt}]
    request_payload = {
        "model": model,
        "messages": message,
        "temperature": 0.8,
        "max_tokens": 3000,
        "frequency_penalty": 0.0
    }
    url="https://api.openai.com/v1/chat/completions"
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",  # replace with your API key
        "Content-Type": "application/json"
    }
    async with aiohttp.ClientSession() as session:
        response = await fetch(session, url, headers, request_payload)
    # print(f"Current scene observation: {node}")print(response)
    # print("=====Error===",response)
    extrapolated_scene = response['choices'][0]['message']['content'].strip()
    print("===Extrapolated scene:", extrapolated_scene)

    return extrapolated_scene


