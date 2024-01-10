import os
import json
import random
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict, HumanMessage

from generative_monster.interface.twitter import TwitterInterface
from generative_monster.generator.openjourney import OpenJourneyGenerator
from generative_monster.generator.leap import LeapGenerator
from generative_monster.prompts import PROMPT_SUFFIXES
from generative_monster.utils.image import open_image, resize_image, save_image
from generative_monster.settings import (
    AGENT_DESCRIPTION,
    HASHTAGS,
    TEMPERATURE,
    GENERATORS_TWITTER_ACCOUNTS
)

class Monster:

    def __init__(self):
        pass


    def create(self, publish=True):
        # Inspiration
        print("-- Memory and inspiration")
        text = self.find_inspiration()
        print("Generated description:", text)
        if len(text) > 200:
            text = text[:190] + "..."
            print("Warning: It was too long! Shortening:", text)
        
        # Appending hashtags
        # tweet_content = text + "\n\n" + HASHTAGS
        # tweet_content = HASHTAGS
        # print("Tweet content:", tweet_content)

        # Deciding on style
        print("--- Style")
        available_styles = list(PROMPT_SUFFIXES.keys())
        selected_style = random.choice(available_styles)
        print("Selected style:", selected_style)

        # Prompt creation
        print("--- Prompt creation")
        prompt = self.create_prompt(text, style=selected_style)
        print("Final prompt:", prompt)

        # Image generation
        print("-- Image generation")
        available_generators = ["openjourney", "leap"]
        selected_generator = random.choice(available_generators)
        print("Selected generator:", selected_generator)
        image_path = self.generate(prompt, generator=selected_generator)
        if not image_path:
            print("Failed to generate image. Please try again later... aborting.")
            return
        print("Generated image:", image_path)
        
        # Validate image
        print("-- Validating image")
        if not self.is_valid(image_path):
            print("Not a valid image. Please try again later... aborting.")
            return
        print("Valid image...")

        # Scale up
        print("-- Scaling image up")
        scale_factor = 2
        image_path = self.scale_image(image_path, scale_factor)
        print(f"Scaled image by x{scale_factor}")

        # Communication
        if publish:
            # generator_twitter = GENERATORS_TWITTER_ACCOUNTS[selected_generator]
            # tweet_content = f"Generated using {generator_twitter} API"
            tweet_content = ""
            print("-- Communication")
            response = self.publish(tweet_content, prompt, [image_path])
            print("Tweet:", response)

        return image_path


    def create_from_prompt(self, initial_prompt, style, generator="openjourney"):
        # Generate image from prompt straight
        prompt = self.create_prompt(initial_prompt, style)
        print("\tPrompt:", prompt)
        image_path = self.generate(prompt, generator)
        print("\tImage:", image_path)
        return image_path


    def find_inspiration(self):
        # TODO Search twitter for daily headlines? Movies? TVSeries?

        # Recover memory
        if os.path.exists("memory.json"):
            # Use existing memory
            with open("memory.json", "r") as f:
                memory_dict = json.load(f)
                messages = messages_from_dict(memory_dict)
                memory = ConversationBufferMemory(return_messages=True)
                # Constraint 
                max_messages = 50
                for message in messages[-max_messages:]:
                    if isinstance(message, HumanMessage):
                        memory.chat_memory.add_user_message(message.content)
                    else:
                        memory.chat_memory.add_ai_message(message.content)
        else:
            # Or create new one
            memory = ConversationBufferMemory(return_messages=True)
        memory.load_memory_variables({})

        # Create a prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(AGENT_DESCRIPTION),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        llm = ChatOpenAI(temperature=TEMPERATURE)
        conversation = ConversationChain(
            memory=memory,
            prompt=prompt,
            llm=llm,
            verbose=False
        )

        gen_prompt = conversation.predict(
            input="Describe a painting in a short phrase, maximum of 10 words, about a topic of your choice. Limit the your answer to 100 characters. Do not quote.")
        
        # gen_text = conversation.predict(
        #     input="Write a tweet about your latest painting to share with your followers. Limit the answer to maximum 100 characters."
        # )

        # Save to memory
        with open("memory.json", "w") as f:
            memory_dict = messages_to_dict(memory.chat_memory.messages)
            json.dump(memory_dict, f)

        return gen_prompt.strip()


    def create_prompt(self, text, style="acrylic"):
        suffix = PROMPT_SUFFIXES[style]["suffix"]
        prompt = text + " " + suffix
        return prompt


    def generate(self, prompt, generator="openjourney"):
        if generator == "openjourney":
            gen = OpenJourneyGenerator()
        elif generator == "leap":
            gen = LeapGenerator()
        image_path = gen.generate(prompt)
        return image_path


    def publish(self, text, prompt, image_paths):
        ti = TwitterInterface()
        res = ti.tweet_with_images(text, prompt, image_paths)
        return res


    def scale_image(self, image_path, scale_factor=2):
        original_image = open_image(image_path)
        resized_image = resize_image(original_image, scale_factor)
        # Overwrite original path for now
        save_image(resized_image, image_path)
        return image_path


    def is_valid(self, image_path):
        # Pitch black images are not valid
        image = open_image(image_path)
        image_array = np.array(image)
        image_mean = image_array.mean()
        return image_mean > 0.0