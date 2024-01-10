import logging
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from constants import LABEL_DESCRIPTION

# for openai logging
# os.environ['OPENAI_LOG'] = 'info'
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv('../.env/secrets.env')
OPENAI_TOKEN = os.getenv('OPENAI_TOKEN')

client = OpenAI(
    api_key=OPENAI_TOKEN
    )

def generate_user_story(label, description, last_stories_for_label):
    previous_stories = "\n".join(last_stories_for_label[-10:])

    user_message = (
        f"Category: {label}\nDefinition: {description}\n"
        f"Here are the last 10 stories for this label:\n{previous_stories}\n"
        "Generate a new user story different from the above, based on the category and definition."
    )
    system_message = ("You are an AI trained in software development and Agile Scrum practices. "
                      "Your task is to generate a concise, clear, and atomic user story that adheres to " 
                      "Agile principles. Each story should focus on a single, actionable item that provides value to the end user. "
                      "The story should be in the format: 'As a [type of user], I want [an action or feature], so that [benefit or value].'"
                      " Ensure the story is specific, measurable, achievable, relevant, and time-bound (SMART), "
                      "and aligns with the principles of the given label.")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            frequency_penalty=1,
            presence_penalty=1
        )
        story_content = response.choices[0].message.content
        return story_content
    except Exception as e:
        logging.error(f"Error generating user story: {e}")
        

last_stories = {label: [] for label in LABEL_DESCRIPTION}

user_stories = []
total_labels = len(LABEL_DESCRIPTION)
stories_per_label = 40 

for story_number in range(1, stories_per_label + 1):
    for index, (label, description) in enumerate(LABEL_DESCRIPTION.items()):
        # Generate the story
        story = generate_user_story(label, description, last_stories[label])
        
        # Add the story to the user stories list and update the last stories tracking
        user_stories.append({'User Story': story, 'Label': label})
        last_stories[label].append(story)
        if len(last_stories[label]) > 10:
            last_stories[label].pop(0)

        remaining_stories = (total_labels - (index + 1)) * stories_per_label + (stories_per_label - story_number)
        logging.info(f"Generated story {story_number} for label '{label}'. Remaining stories: {remaining_stories}")



df = pd.DataFrame(user_stories)
df.to_csv('generated_user_stories.csv', index=False)