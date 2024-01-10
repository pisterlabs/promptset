import openai
import os
import json

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')
script_dir = os.path.dirname(os.path.abspath(__file__))
stories_file = os.path.join(script_dir, '../data/stories.json')

prompt = '''d
            You are going to create short stories (about 250 words each) for children that introduce concepts of logic and programming in a fun and engaging way. And in the end a simple question that every child should be able to answer.\n
            These stories can feature characters and adventures that incorporate logical thinking, problem-solving, and basic programming concepts in a way that's easy for kids to understand and enjoy.\n
            For each of the following categories, create a story that allows a child to learn about the topic: \n
                Sequences\n
                Conditional Statements\n
                Loops and Repetition\n
                Variables\n

            The format of your response should be an array of json objects, where each object contains the following properties: \n
                title: The title of the story\n
                text: The text of the story\n
                category: The category of the story

            For example: \n
                [
                    {
                        "title": "The Three Little Pigs",
                        "text": "Once upon a time, there were three little pigs. The first little pig built his house out of straw. The second little pig built his house out of sticks. The third little pig built his house out of bricks. One day, a big bad wolf came to the first little pig's house and said, \"Little pig, little pig, let me in.\" The little pig said, \"Not by the hair of my chinny chin chin.\" The big bad wolf said, \"Then I'll huff and I'll puff and I'll blow your house down.\" So he huffed and he puffed and he blew the house down. The first little pig ran to the second little pig's house. The big bad wolf came to the second little pig's house and said, \"Little pig, little pig, let me in.\" The little pig said, \"Not by the hair of my chinny chin chin.\" The big bad wolf said, \"Then I'll huff and I'll puff and I'll blow your house down.\" So he huffed and he puffed and he blew the house down. The first and second little pigs ran to the third little pig's house.",
                        "category": "Sequences"
                    }
                ]
        ''' 

"""Returns a list of stories from OpenAI's API."""
def get_daily_stories():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ],
        temperature=1,
        max_tokens=3350,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    
    stories = response.choices[0].message["content"]
    
    # Convert the stories from a string to a list of json objects
    stories = json.loads(stories)

    # Add new stories to the existing ones (on stories.json)
    with open(stories_file, 'r') as f:
        stories_json = json.load(f)
        curr_num_stories = len(stories_json)

        # Add new field id to each story
        for i, story in enumerate(stories):
            new_id = curr_num_stories + i + 1
            story["id"] = new_id
            story["audio"] = f"https://codetales.blob.core.windows.net/code-tales/story-{new_id}.mp3"
            story["illustration"] = [f"https://codetales.blob.core.windows.net/code-tales/story-{new_id}.jpg"]
            
        stories_json.extend(stories)

    print(stories_json)

    # Save the updated stories to the stories.json file
    with open(stories_file, 'w') as f:
        json.dump(stories_json, f)

    return stories

def get_stories():
    with open(stories_file, 'r') as f:
        return json.load(f)
    
#print(get_stories())

#stories = get_daily_stories()
#print(stories)
