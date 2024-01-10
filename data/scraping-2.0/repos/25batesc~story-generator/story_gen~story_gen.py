import time
import openai
from openai import error as openai_error
import os
import json
import regex
exceptions = (
    openai_error.Timeout, 
    openai_error.APIError, 
    openai_error.APIConnectionError, 
    openai_error.InvalidRequestError, 
    openai_error.AuthenticationError, 
    openai_error.PermissionError, 
    openai_error.RateLimitError,
)
import threading

class StoryGenerator:
    def __init__(self, ai_model="gpt-3.5-turbo"):
        self.ai_model = ai_model
        self.system_role = '''You are a legendary storywriter who writes very brief but powerful and unique story summaries.'''
        self.version = "0.1"
        '''You are famous for your Story Circle way of forming narratives.
The Steps Explained
1. In A Zone Of Comfort
In the first step, the protagonist is surrounded by a world known to them, where they are in control of their situation. This world is unchallenging and the protagonist lives a relatively mundane everyday.
2. They Desire Something
The protagonist really wants something. They want to achieve this goal so bad they will go to great lengths to achieve it. However, this desire is out of their reach and throws them out of their comfort zone.
3. Enter An Unfamiliar Situation
In order to achieve this goal or desire the protagonist has to enter unknown territory. They are thrown into a world beyond their control.
4. Adapt To The Situation
The protagonist combines their already established skills with their newly acquired skills to fully adapt to their new surroundings. However, this takes time which can lead to trouble as time is never on their side.
5. Get What They Desired
The one thing they truly wanted is gained but other obstacles follow close behind.
6. Pay A Heavy Price For Winning
When things go too well bad things start to happen. The protagonist wins something but loses another thing. Something important or meaningful to the protagonist has been lost.
7. A Return To Their Familiar Situation
The protagonist returns to their normal world. As a result, they ease back into their zone of comfort, where everything is familiar again.
8. They Have Overall Changed
However, after entering back into their familiar world, the protagonist does not return as the same person. A deep-rooted trait has changed inside them, whether that be a fear they have overcome or a character flaw that they have changed. Although, by the end of the journey the character’s everyday life has been enriched by their experience.'''
        self.story_circle_steps = [
"Our character starts in a familiar, comfortable environment.",
"Feeling a desire or need, the character wants something they don't currently have.",
"To obtain what they want, the character leaves their comfort zone and enters an unfamiliar situation or world.",
"In this unfamiliar situation, the character adapts and learns new skills to survive.",
"After facing challenges, the character finally gets what they wanted.",
"However, getting what they wanted comes with a heavy price, resulting in unforeseen consequences.",
"After paying the price, the character returns to their familiar situation or world.",
"But they return having changed, grown, or evolved from their journey."
        ]

    def extract_json_string(self, text):
        pattern = r"\{(?:[^{}]|(?R))*\}"
        result = regex.search(pattern, text, regex.DOTALL)
        if result:
            return result.group(0)
        else:
            print(f"No JSON string found in response: {text}")
            return None

    def gpt_request(
        self,
        prompt,
        system_role,
        model="gpt-3.5-turbo",
        temperture=.99,
        enforce_parseable=False,
        content_only=True,
        max_tokens=3000,
        max_retries=10,
    ):
        retries = 0

        while retries < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_role,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperture,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )

                result = response
                message_content = response["choices"][0]["message"]["content"]

                if enforce_parseable:
                    json_body = self.extract_json_string(message_content)
                    if json_body is None:
                        print(
                            f"{model} Non-Parseable Response Recieved: {message_content}"
                        )
                        time.sleep(2**retries)
                        retries += 1
                        continue
                    try:
                        message_json = json.loads(json_body)
                        if content_only:
                            return message_json
                        else:
                            return response
                    except json.decoder.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        time.sleep(2**retries)
                        retries += 1
                        continue

                if content_only:
                    return message_content
                return result
            except exceptions as e:
                print(f"Error: {e}")
                time.sleep(2**retries)
                retries += 1
        # return None
        raise ConnectionError("Max retries reached. Request failed.")

    def write_file(self, folder_path, base_name, content, extension="txt"):
        os.makedirs(folder_path, exist_ok=True)
        
        count = 0
        filename = f"{folder_path}/{base_name}_{self.version}.{extension}"
        
        # Check if the file already exists, and append a count if it does
        while os.path.exists(filename):
            count += 1
            filename = f"{folder_path}/{base_name}_{self.version}_{count}.{extension}"

        with open(filename, "w") as file:
            file.write(content)

    def generate_story(self, initial_prompt, story_name):
        story_circle_structure = self.generate_story_circle(initial_prompt)
        folder_path = f"story/{story_name}"
        
        content = ""
        step_number = 1
        for step in story_circle_structure:
            content += f"Step {step_number}:\n"
            for scene in step:
                content += f"    {scene}\n"
            content += '\n'
            step_number += 1

        base_name = f"story_circle_{story_name}"
        # Call the common write function to save the story
        self.write_file(folder_path, base_name, content)
        quit()

        context = {}  # Initialize your context object
        scenes = self.generate_scenes(story_circle_structure, context)

        for i, scene in enumerate(scenes):
            with open(f"{folder_path}/scene_{i}.txt", "w") as file:
                file.write(scene)
                
        return scenes

    def generate_story_circle(self, initial_prompt):
        # Initialize the Story Circle structure
        story_circle_structure = []

        # Iterate over each step in the Story Circle
        for i, step in enumerate(self.story_circle_steps, 1):
            # Add the Story Circle context up to this point to the prompt
            story_circle_context = ' '.join(story_circle_structure)
            prompt = f'''{initial_prompt}
{i-1} steps of the Story Circle have been completed already;
Complete the next step of the Story Circle, Step {i}:'{step}';
Generate a key event or decision inline with '{step}';
Review the completed Story Circle steps with our goals in mind [[[{story_circle_context}]]];
Now, instead of summarizing, describe the scenes that comprise '{step}'. 
Consider the senses - what do the characters see, hear, feel, smell, taste? 
What actions do they take? What is their emotional response? 
Describe the scenes as if they are happening in real time, but be succicnt as your response is limited to 300 tokens.
Format:
Step {i}:
##scene_num:[scene_description]##
##scene_num:[scene_description]##'''

           
            # Generate the event for this step of the Story Circle
            event = self.gpt_request(prompt, self.system_role, self.ai_model, max_tokens=600)

            # Add the event to the Story Circle structure
            story_circle_structure.append(event)

            '''# Print the Story Circle so far
            print(f"Story Circle up to step {i}:")
            for j, event in enumerate(story_circle_structure, 1):
            print(f"Step {j}: {event}")
            print("\n")'''
            print(f"\n{event}")
            # Wait a bit to prevent hitting rate limit
            time.sleep(1)

        return story_circle_structure

    def generate_scenes(self, story_circle_structure, context):
        scenes = []
        # For each point in the Story Circle...
        for point in story_circle_structure:
            # Create a prompt for the AI model
            prompt = f"Generate a scene for the following point in the Story Circle: {point}. Context: {context}"
            # Use the model to generate a scene
            scene = self.gpt_request(prompt, self.system_role)
            # Add the scene to the list of scenes
            scenes.append(scene)
            # Update the context for the next scene
            context = self.update_context(context, scene)
            time.sleep(1)  # To prevent hitting rate limit
        return scenes

    def update_context(self, context, scene):
        # Extract details from the scene
        # This is just an example, you would need to decide what details to extract based on your requirements
        characters = extract_characters(scene)
        setting = extract_setting(scene)
        plot_details = extract_plot_details(scene)

        # Add these details to the context
        context["characters"] = characters
        context["setting"] = setting
        context["plot_details"] = plot_details

        return context

    def generate_story_ideas(self, all_story_prompt_prefix = ''):
        # Prompt to ask GPT for 5 interesting story ideas
        idea_prompt = f'''
We need 5 unique and engaging story ideas.
Format the ideas in python parseable json.
Example of one idea - the keys and values:
###
Instruction: You will edit a story board and add to it with cooky characters and unique narrative threads;
Genre: Sinister Body Horror,
Title: Nueromaggot,
Description: A criminal is on the run, and is being hunted by a futuristic cybercorp that is hell bent on destroying him.
###
Return an array of 5 dictionaries with the keys from the example, each representing a unique and distinct story idea.
'''

        # Request the ideas from GPT
        idea_response = self.gpt_request(idea_prompt, self.system_role, model='gpt-4',max_tokens=1500, enforce_parseable=False)
        idea_list = json.loads(idea_response)
        print(idea_response)
        threads = []
        # Iterate through the ideas, extracting relevant details and generating stories
        for idea in idea_list:
            print(idea)
            instruction = idea["Instruction"]
            genre = idea["Genre"]
            title = idea["Title"]
            description = idea["Description"]

            # Construct the content to save
            content = f'''Instruction: {instruction}
    Genre: {genre}
    Title: {title}
    Description: {description}'''

            # Create a folder path
            folder_path = f"story/{title}"
            print(content)
            # Call the write_file method to save the idea
            self.write_file(folder_path, f"story_idea_{title}", content)

            # Construct the prompt for story generation
            prompt = f'''{all_story_prompt_prefix}
    Instruction:{instruction}
    Genre: {genre}
    Title: {title}
    Description: {description}'''


            # Create a new thread for each story generation
            thread = threading.Thread(target=self.generate_story, args=(prompt, title))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return [idea["Title"] for idea in idea_response]

# Usage
story_generator = StoryGenerator()

ideas = story_generator.generate_story_ideas(all_story_prompt_prefix='')
print(ideas)

"""story_name = "Nueromaggot"
story = story_generator.generate_story(
    f'''Instruction: You will edit a story board and add to it with cooky characters and unique narrative threads;
    Genre: Sinister Body Horror;
    Title: {story_name}''', story_name)
print(story)"""