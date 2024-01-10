import time
import openai
from openai import error as openai_error

class StoryGenerator:
    def __init__(self, ai_model="gpt-4"):
        self.ai_model = ai_model
        self.system_role = "You are a legendary unnamed storywriter."
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
        self.story_circle_steps_lite = [
"A character is in a zone of comfort.",
"They desperately want something.",
"They enter an unfamiliar situation.",
"They adapt to that unfamiliar situation.",
"They get what they wanted after much effort.",
"Knowingly or unknowingly, they pay a heavy price.",
"They return back to their zone of comfort.",
"They’ve been changed forever."
        ]

    def gpt_request(
        self,
        prompt,
        system_role,
        model="gpt-3.5-turbo",
        temperture=0.98,
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

            except openai_error.Timeout as e:
                # Handle timeout error, e.g. retry or log
                print(f"OpenAI API request timed out: {e}")
                time.sleep(2**retries)
                retries += 1

            except openai_error.APIError as e:
                # Handle API error, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(2**retries)
                retries += 1

            except openai_error.APIConnectionError as e:
                # Handle connection error, e.g. check network or log
                print(f"OpenAI API request failed to connect: {e}")
                time.sleep(2**retries)
                retries += 1

            except openai_error.InvalidRequestError as e:
                # Handle invalid request error, e.g. validate parameters or log
                print(f"OpenAI API request was invalid: {e}")
                time.sleep(2**retries)
                retries += 1

            except openai_error.AuthenticationError as e:
                # Handle authentication error, e.g. check credentials or log
                print(f"OpenAI API request was not authorized: {e}")
                time.sleep(2**retries)
                retries += 1

            except openai_error.PermissionError as e:
                # Handle permission error, e.g. check scope or log
                print(f"OpenAI API request was not permitted: {e}")
                time.sleep(2**retries)
                retries += 1

            except openai_error.RateLimitError as e:
                # Handle rate limit error, e.g. wait or log
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(2**retries)
                retries += 1
        # return None
        raise ConnectionError("Max retries reached. Request failed.")

    def generate_story(self, initial_prompt):
        story_circle_structure = self.generate_story_circle(initial_prompt)
        #print(story_circle_structure)
        context = {}  # Initialize your context object
        quit()
        scenes = self.generate_scenes(story_circle_structure, context)
        return scenes

    def generate_story_circle(self, initial_prompt):
        # Initialize the Story Circle structure
        story_circle_structure = []

        # Iterate over each step in the Story Circle
        for i, step in enumerate(self.story_circle_steps, 1):
            # Add the Story Circle context up to this point to the prompt
            story_circle_context = ' '.join(story_circle_structure)
            prompt = f'''{initial_prompt}. So far, the story has unfolded as follows: {story_circle_context}.
Now, focusing only on the following stage of the Story Circle - '{step}' - generate a key event or decision to move the story forward in unnexpected but satisfying ways.
Format: ##scene_num. [scene_description]##'''

            # Generate the event for this step of the Story Circle
            event = self.gpt_request(prompt, self.system_role,max_tokens=500)

            # Add the event to the Story Circle structure
            story_circle_structure.append(event)

            # Print the Story Circle so far
            print("\n")
            print(f"Story Circle up to step {i}:")
            print(f"\n{event}\n")
            

            # Wait a bit to prevent hitting rate limit
            time.sleep(.5)

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


# Usage
story_generator = StoryGenerator()
story = story_generator.generate_story("Genre: Action Thriller; Title: Regala Extrema;")
print(story)
