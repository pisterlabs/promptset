import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_story(topic=None):
    print("Generating story...")
    
    prompt = """
        Generate a compelling personal narrative that simulates a story one might share in profound conversation. The delivery should feel candid and authentic, as if recounted by an ordinary individual about a significant episode in their life. The language can be informal, mirroring everyday dialogue.

        Adhere to the protagonist's gender provided. 

        The story must tackle an intriguing or challenging topic—something more profound than the run-of-the-mill life experiences. Think of scenarios that might spark lively debates on platforms like AITA on Reddit, or narratives that tug at heartstrings, culminating in an unexpected turn of events.

        Guideline for your narrative:

        - The topic should incite curiosity and engagement.
        - The narrative should be captivating and unique, far from mundane.
        - Avoid personal interjections, let the story unfold by itself.
        - Initiate with an engaging, casual title like, "How I narrowly... " or "Why I'll never again... "
        - Craft the narrative to feel intimate and immediate, akin to a gripping short story on a Reddit thread.
        - Don't include summaries or explanations at the end. You may conclude with a brief one-liner reaction, if desired.
        - Title should be crafted as a complete sentence.

        Please format your response in JSON with the properties 'title', 'content', 'gender' (either 'male' or 'female'), and 'description'. Ensure to escape the quotes by adding a backslash before them. For instance, if your title is "How I narrowly... ", it should be formatted as \"How I narrowly... \". Refrain from using newline characters such as \n.
    """

    # Check if a topic is provided and append to paragraph if true
    if topic:
        prompt += f"\nBase the story off this topic: {topic}"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )

    # Extract the generated story
    story = completion.choices[0].message.content

    # Parse the JSON output
    story_dict = json.loads(story)

    # Extract the title, content, and description
    title = story_dict.get("title")
    content = story_dict.get("content")
    description = story_dict.get("description")
    gender = story_dict.get("gender").lower()

    # Remove escape characters as well as \n and \t
    title = title.replace("\\", "").replace("\n", " ").replace("\t", "")
    content = content.replace("\\", "").replace("\n", " ").replace("\t", "")
    description = description.replace("\\", "").replace("\n", " ").replace("\t", "")

    return title, content, description, gender
