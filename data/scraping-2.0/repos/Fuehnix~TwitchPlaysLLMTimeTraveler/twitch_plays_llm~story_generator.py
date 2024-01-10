import asyncio
from loguru import logger
import openai

from asgiref.sync import sync_to_async

from .misc import log_exceptions

from .models import StoryEntry


class StoryGenerator:
    def __init__(self):
        # TODO: Dynamically generate initial prompt
        initial_entry = StoryEntry(
            story_action='',
            # narration_result="You are a middle aged man in downtown Chicago, 1910. You're in a steak restaurant talking to the waiter as you just sat down.",
            # narration_result="You are a quirky time travelling inventor with a handlebar mustache and a knack for mischievous inventions. Blinking your eyes open, you realize you have accidentally landed in the year 1875, right in the heart of a bustling Wild West town. Dusty roads, saloons, and cowboys on horseback surround you, while the sound of piano music drifts through the air.",
            narration_result=
            """Welcome, brave Esther, to the sprawling city of Gearlock, a symphony of cogwheel and steam where airships drift through the sooty skies and giant gearworks define the horizon. Here, a bird's-eye view is a literal commodity you possess, becoming as sparrows or falcons at will. Time isn't a river but a swirling eddy for you, bendable and controllable. The diverse, lively chatter of your "Twitch" keeps your world kaleidoscopic, pushing and pulling you through the cacophonic rhythm of your dual existence. As you walk through the vibrant brass streets, your skilled eyes see the intricate beauty of life sketched in every corner. A sudden flutter of wings catches your attention; a mechanical messenger pigeon lands near you, a note gripped in its tiny metallic talons. A quick scan of the message, and it's clear: an urgent summons from the enigmatic Clockwork Guildmaster, a call to action that your many voices are eager to answer."""
        )
        self.past_story_entries = [
            initial_entry
        ]
        self.generate_image_task = asyncio.create_task(self._generate_narration_image(initial_entry))

    def construct_initial_prompt(self):
        """Not used
        construct initial prompt for story generation"""
        # rules = """Create a writing prompt to start an RPG text adventure game.  Adhere to the following rules:
        #             1. The story should take place in Baldur's Gate from Dungeons and Dragons' Forgotten Realms.
        #             2  You should describe the player's characteristics, where they are, what time period they are in, and what surrounds them.
        #             3. Keep it fun and allow people to be creative.
        #             4. Use the 2nd person perspective.
        #             5. The prompt should be only 3 - 5 sentences long."""
        rules = """Please complete the following story prompt by describing the start of a Steampunk setting for a text adventure game:
                Please keep your response
                1. limited to a paragraph 
                2. write in the style of Critical Role D&D
                3. Include a vague call to action for the player to act on, such as a messenger having a quest, seeing someone in an alley, someone pleads for help, you encounter an unusual but mundane item, etc.

                Story prompt: 
                You are a human woman named Esther, a friendly bird obsessed artist and programmer. You have the power to transform into a bird and have time related abilities.  You have many voices in your head that control your actions, sometimes you call them your “Twitch”. """
        messages = [{ 'role': 'user',
                'content': rules}]

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages = messages,
        )
        initial_prompt = response['choices'][0]['message']['content']
        print('generated initial prompt')
        return initial_prompt

    def construct_prompt_messages(self, story_action: str):
        # === ChatCompletions API reference ===
        # system: tells ChatGPT what it's role is/the context of its responses
        # assistant: pseudo-history of messages from openai model
        # user: pseudo-history of messages from user
        #
        # Bot will then try to complete the conversation
        messages = [
            {
                'role': 'system',
                # 'content': 'You are a storyteller that generates the next event in a story based on the action a user says. The story takes place in 1910 and the main character is a middle aged man. At each turn, the user says an action and you reply with a short continuation of the story outlining the events that happen in the story based on the action the user performed.',
                'content': """Please perform the function of a text adventure game, following the rules listed below:

                            Presentation Rules:

                            1. At each turn, the user says an action and you reply with a short continuation of the story outlining the events that happen in the story based on the action the user performed.

                            2. Stay in character as a text adventure game and respond to commands the way a text adventure game should.

                            Fundamental Game Mechanics:

                            1. If an action is unsuccessful, respond with a relevant consequence.

                            2. Allow players to be creative, and indulge them in their distractions.

                            3. The voices in Esther's head can compel her to do anything.

                            Refer back to these rules after every prompt.

                            Start Game.""",
            },
        ]
        for story_entry in self.past_story_entries:
            if story_entry.story_action:
                messages += [{'role': 'user', 'content': story_entry.story_action}]
            if story_entry.narration_result:
                messages += [
                    {
                        'role': 'assistant',
                        'content': story_entry.narration_result,
                    }
                ]
        messages.append({'role': 'user', 'content': story_action})
        return messages

    async def generate_next_story_narration(self, story_action: str) -> StoryEntry:
        entry = await self._generate_next_story_narration(story_action)
        if self.generate_image_task:
            await self.generate_image_task
            self.generate_image_task = asyncio.create_task(self._generate_narration_image(entry))
        return entry

    @sync_to_async
    def _generate_next_story_narration(self, story_action: str) -> StoryEntry:
        """Generates the continuation of the story given a user action"""
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages=self.construct_prompt_messages(story_action),
        )
        next_narration = response['choices'][0]['message']['content']
        entry = StoryEntry(story_action=story_action, narration_result=next_narration)
        self.past_story_entries.append(entry)
        return entry

    @sync_to_async
    @log_exceptions
    def _generate_narration_image(self, story_entry: StoryEntry):
        """Populate the narration_image_url of the provided story entry using OpenAI image API"""
        logger.debug('Generating image caption...')
        story_prefix = self.past_story_entries[0].narration_result[:500] + '...\n'
        if len(self.past_story_entries) == 1:
            story_prefix = ''
        story_summary = story_prefix + self.past_story_entries[-1].narration_result
        image_caption = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': 'Write a story.'},
                {'role': 'assistant', 'content': story_summary},
                {'role': 'user', 'content': 'Think of an image that depicts the world of this story, focusing on the most recent event. Write a caption of this image (ie. a series of fragment descriptors). The sentence format and length should be similar to this example: "Cyberpunk digital art of a neon-lit city with a samurai figure, highlighting the contrast between traditional and futuristic".'}
            ],
        )['choices'][0]['message']['content']
        logger.info('Generated image caption: {}', image_caption)
        logger.debug('Generating image...')
        image_url = openai.Image.create(
            prompt=image_caption,
            n=1,
            size="1024x1024"
        )['data'][0]['url']
        logger.info('Generated image: {}', image_url)
        story_entry.narration_image_url = image_url

    @sync_to_async
    def generate_image_prompt(self):
        """Generates a prompt for DALL-E based on the current scene"""
        # Use the last narration result as the scene description
        scene_description = self.past_story_entries[-1].narration_result
        return scene_description

    def reset(self):
        initial_entry = StoryEntry(
            story_action='',
            # narration_result="You are a middle aged man in downtown Chicago, 1910. You're in a steak restaurant talking to the waiter as you just sat down.",
            # narration_result="You are a quirky time travelling inventor with a handlebar mustache and a knack for mischievous inventions. Blinking your eyes open, you realize you have accidentally landed in the year 1875, right in the heart of a bustling Wild West town. Dusty roads, saloons, and cowboys on horseback surround you, while the sound of piano music drifts through the air.",
            # narration_result="""In the heart of the iron-clad city of Gearford, within the cloud-shrouded aeries of the Cog Tower, you, Esther, find solace among the thrumming machinations and whistling steam pipes, your fingers dancing across the canvas and keyboard alike. From the corner of your eye, you witness the blinking gears of your ornithopter clock, its rhythmic tick-tocking a constant reminder of your temporal prowess. Yet, the whispering voices in your mind, your loyal Twitch, sing in discordant harmony, guiding, prodding, or sometimes even commanding you. As you shape-shift into a shimmering bird and take flight, the metropolis sprawls beneath you, a mechanical marvel of brass and steam. Below, in the twisting alleyways, you catch sight of a frantic messenger being accosted by clockwork constables, his desperate eyes seemingly pleading for your intervention. It seems that Gearford, once again, requires the touch of your wing and the turn of your gear."""
            narration_result= self.construct_initial_prompt(),
        )
        self.past_story_entries = [initial_entry]
        self.generate_image_task = asyncio.create_task(self._generate_narration_image(initial_entry))
