import openai
import os
import random
import json

# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define a function to generate a story
def generate_story(audience, story_type, plot_twist_type):
    while True:
        # Define the prompt for the GPT-3 API
        prompt = f"You are the narrator of a video short for {audience}, write it in only 5 paragraphs, each paragraph should be 25 words or less, but not less than 5. Narrate a {story_type} story with a {plot_twist_type} plot twist. Do not include opening or closing remarks. Do not mention scenes, parts or any storyboard technical aspect, do not start each paragraph with a paragraph number: enumeration. remember to split your response in five different separated paragraphs."

        # Call the GPT-3 API to generate the story
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
            n=1,
            stop=None,
            timeout=15,
        )

        # Extract the story from the response
        story = response.choices[0].text.strip()

        # Split the story into 5 paragraphs
        paragraphs = story.split("\n\n")
        
        # Check if the story has 5 paragraphs
        if len(paragraphs) == 5:
            # Check if each paragraph has at least 5 words
            if all(len(paragraph.split()) >= 5 for paragraph in paragraphs):
                return paragraphs

# Define the possible values for each parameter
audiences = ["kids", "hipsters", "businessmen", "dogs","cats","capybaras","everybody","babies","teens", "adults", "seniors", "zombies", "ants", "housewives", "sailors", "pirates", "hippies", "nerds", "yuppies", "boomers", "millenials", "centennials"]
story_types = ["uplifting", "dark", "love", "fun", "action", "adventure", "scary", "thriller", "mystery", "Romance", "Parody", "Comical", "funny", "sci-fi", "fantasy", "Historical", "Biographical", "Documentary", "weird"]
plot_twist_types = ["unexpected", "dark", "funny", "wholesome", "surprising", "horrifying", "disturbing", "genius", "expected", "clever", "hysterical", "epic", "blunt", "sharp", "brilliant", "spectacular", "smart"]

# Select random values for each parameter
audience = random.choice(audiences)
story_type = random.choice(story_types)
plot_twist_type = random.choice(plot_twist_types)

# Generate the story
paragraphs = generate_story(audience, story_type, plot_twist_type)
tags = audience, story_type, plot_twist_type


# Print the title and the paragraphs
title = f"{story_type} story for {audience} with a {plot_twist_type} plot twist"
print(title)
print("--------------------------------------------------------------------------------------------------")
# Print each paragraph on a separate line
for i, paragraph in enumerate(paragraphs):
    print(paragraph)

# Save the story as a JSON file
story_dict = {'Title': title, 'Paragraphs': paragraphs, 'Tags': tags,}
with open('story.json', 'w') as f:
    json.dump(story_dict, f, indent=4)
