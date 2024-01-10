import json
import os
import sys

from VNImageGenerator import VNImageGenerator

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class CharacterBuilder:
    def __init__(self, chat_llm=None):
        if chat_llm is None:
            self.chat_llm = ChatOpenAI(
                model_name='gpt-4',
                temperature=0.5
            )
        else:
            self.chat_llm = chat_llm

    example_prompt_json = '''
        {
          "name": "Sakura Nadeshiko",
          "character_base": "girl",
          "age": "18 years old",
          "hair": "short blond hair",
          "eyes": "blue eyes",
          "face": "pretty childlike face, detailed face",
          "body": "slim, slender, petite, small chest",
          "wearing": "traditional japanese clothing, pink kimono, yukata"
          "personality": "kind, caring, cheerful, optimistic, outgoing, friendly, energetic, playful, mischievous"
          "backstory": "Sakura grew up in a small town in Japan. She was always a cheerful and optimistic child, and she loved to play with her friends. She now attends a prestigious high school in Tokyo "
          "motivations": "Sakura is motivated by her desire to help others. She wants to make the world a better place, and she believes that she can do this by becoming a doctor. She also wants to make her parents proud of her."
        }
        '''

    prompt_format_json = '''
            {
              "name": <the character's name>,
              "character_base": <boy, girl, man, woman, etc.>,
              "age": "<age in years or description of age>",
              "hair": <description of hair color and style>,
              "eyes": "<eye color, and other description of eyes>",
              "face": "<description of face>",
              "body": "<description of body>",
              "wearing": "<a detailed description of clothing worn by the character and its colors>",
              "personality": "<a detailed description of the character's personality>",
              "backstory": "<a detailed description of the character's backstory>",
              "motivations": "<a detailed description of the character's motivations>"
            }
            '''

    def generate_individual_character(self, prompt, background_information=None):
        # This is the chat model that will be used to generate the structured output

        system = "You are an character designer for several successful fictional franchises. Your stories are known " \
                 "for their complex, lifelike, and believable characters. You are tasked with detailing all characters " \
                 "and their attributes." \
                 "For individual characters, include full descriptions of their physical appearance, " \
                 "their personality, their backstory, and their motivations for their actions. " \
                 "Please be creative, descriptive yet short (separate multiple elements with commas), the stable diffusion " \
                 "engine will use these to " \
                 "generate images and the writers" \
                f"will use these to write the story. You know the following about the character: {prompt}"

        prompt = f"""Create a new character for a visual novel, be creative! Here the output format: 
        {CharacterBuilder.prompt_format_json} 

        Here is an example of the output format (dont follow this exactly, just use it as an example):
        {CharacterBuilder.example_prompt_json}
        
        Also, be careful describing green clothing and eyes, as the artist will use a green screen to draw the character.
        """

        # prepend the background information if it is provided
        if background_information is not None:
            prompt = f"""{background_information}\n\n{prompt}"""

        message = self.chat_llm(
            [
                SystemMessage(role="StoryMaker", content=system),
                HumanMessage(content=prompt),
            ]
        )

        formatted_json = message.content.strip()

        # Attempt to parse the string response into a JSON object
        try:
            structured_output = json.loads(formatted_json)
        except json.JSONDecodeError:
            # Handle the case where the response is not in proper JSON format
            structured_output = "The AI's response was not in a valid JSON format. Please check the AI's output."

        return structured_output

if __name__ == "__main__":
    from util import load_secrets
    load_secrets()

    characters_so_far = []

    # Test the CharacterBuilder class
    cb = CharacterBuilder()
    pipeline_path = "SDCheckpoints/aingdiffusion_v13.safetensors"
    upscaler_model_id = "stabilityai/sd-x2-latent-upscaler"
    generator = VNImageGenerator(pipeline_path, upscaler_model_id)

    # generate 5 characters
    for i in range(15):
        prompt = cb.generate_individual_character("Surprise me come up with a fun new cute anime character. Here are the characters so far: " + str(characters_so_far))

        # save the output to a file
        with open("character.json", "w") as f:
            json.dump(prompt, f)

        # append the whole prompt to the list of characters
        characters_so_far.append(prompt)

        # prettify the output
        print(json.dumps(prompt, indent=4))
        generator.generate_character_images(prompt, save_intermediate=True)


