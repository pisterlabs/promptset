from openai import OpenAI
from config import settings


class WriterGPT:
    """
    A class for generating and refining high fantasy stories using the OpenAI GPT model.

    Attributes:
        client (OpenAI): An instance of the OpenAI API client.

    Methods:
        generate_story(critique, character, world_building, previous_story, critique_turn): Generates a new story or refines an existing one based on provided inputs.
        prompt_template(critique, character, world_building, previous_story, critique_turn): Constructs the appropriate prompt for story generation or refinement.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.API_KEY)

    def generate_story(
        self,
        critique: str | None = None,
        character: str | None = None,
        world_building: str | None = None,
        previous_story: str | None = None,
        critique_turn: bool = False,
    ) -> str:
        """
        Generates a high fantasy story or refines an existing story based on critique, character development, and world-building inputs.

        Args:
            critique (str | None): The critique of the previous story version, used for refinement.
            character (str | None): Character development details for enhancing the story.
            world_building (str | None): World-building details for enriching the story's setting.
            previous_story (str | None): The previous version of the story, if available for refinement.
            critique_turn (bool): Flag to indicate whether the story generation is based on critique.

        Returns:
            response story content (str): The generated or refined story content.
        """
        response = self.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_template(
                        critique=critique,
                        character=character,
                        world_building=world_building,
                        previous_story=previous_story,
                        critique_turn=critique_turn,
                    ),
                }
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def prompt_template(
        self,
        critique: str | None = None,
        character: str | None = None,
        world_building: str | None = None,
        previous_story: str | None = None,
        critique_turn: bool = False,
    ) -> str:
        """
        Constructs a prompt for the GPT model to generate or refine a story based on the provided parameters.

        Args:
            critique (str | None): The critique to be incorporated into the prompt for story refinement.
            character (str | None): Character details to be included in the story generation or refinement.
            world_building (str | None): World-building details to enrich the story's setting in the prompt.
            previous_story (str | None): The previous story text to be considered for refinement.
            critique_turn (bool): Indicates if the prompt is for story refinement based on critique.

        Returns:
            prompt (str): A formatted prompt string for the GPT model.
        """

        if critique_turn:
            after_critquie_prompt = f"""
            Based on the critique and suggestions provided, revise and enhance the high fantasy story.
            Focus on addressing the identified areas of improvement in narrative structure, character development,
            and world-building. Incorporate the suggested changes to enhance plot coherence, deepen character arcs,
            and enrich the descriptive elements of the fantasy world. Ensure the story maintains a balanced flow,
            with improved dialogue, action, and descriptive passages. Pay particular attention to refining the emotional depth 
            and thematic resonance of the narrative. Work on crafting a more engaging and immersive storytelling experience that
            captivates the reader from beginning to end, Write as Markdown, write 4000 tokens.

            critique: {critique}

            character: {character}

            world-building: {world_building}

            keep in mind this is your previous writing: {previous_story}
            """
            return after_critquie_prompt

        init_story_prompt = f"""
        Write a compelling high fantasy story based with the title {settings.STORY_TITLE} and with the following description
        {settings.STORY_DESCRIPTION} focus on the initial plot and character outlines provided.
        Focus on creating a multi-layered narrative with detailed descriptions, complex characters, and an engaging plot structure.
        Ensure the story unfolds with a captivating introduction, develops tension and conflict in the middle, and concludes
        with a satisfying climax. Pay attention to crafting vivid scenes that bring the fantasy world to life, highlighting
        its unique elements, landscapes, and creatures. I want you to write just the first chapter and go on details on it.
        
        here are the details of the characters: {character}
        here are the details of the world: {world_building}
        
        Write as Markdown, write 4000 tokens.
        """
        return init_story_prompt


class CritiqueGPT:
    """
    A class for generating critiques of high fantasy stories using the OpenAI GPT model.

    Attributes:
        client (OpenAI): An instance of the OpenAI API client.

    Methods:
        generate_critique(story): Generates a critique for a given high fantasy story.
        prompt_template(story): Constructs the prompt used for generating the critique.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.API_KEY)

    def generate_critique(self, story: str) -> str:
        """
        Generates a critique for the provided high fantasy story.

        Args:
            story (str): The high fantasy story to be critiqued.

        Returns:
            str: The generated critique, offering insights and suggestions for improvement.
        """
        response = self.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_template(story=story),
                }
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def prompt_template(self, story: str) -> str:
        """
        Generates a prompt template for the provided high fantasy story.

        Args:
            story (str): The high fantasy story.

        Returns:
            str: The generated prompt template.
        """
        return f"""
        Provide a comprehensive critique of the provided high fantasy story with title {settings.STORY_TITLE} 
        and a desciption of {settings.STORY_DESCRIPTION} Analyze the narrative for structure, 
        character development, world-building accuracy, and thematic depth. Identify areas where the plot might lack coherence,
        or character motivations could be clearer. Offer specific suggestions for enhancing emotional depth, narrative complexity,
        and reader engagement. Evaluate the balance of dialogue, descriptive passages, and action, providing recommendations for 
        improvement. Highlight areas where the story might benefit from more detailed descriptions of the fantasy world.
        You will be given just the first chapter of it so you can focus on it. 
        story: {story}
        """


class CharacterGPT:
    def __init__(self):
        self.client = OpenAI(api_key=settings.API_KEY)

    def generate_characters(self) -> str:
        response = self.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_template(),
                }
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def prompt_template(self) -> str:
        return f"""
        Generate characters for the high fantasy story with title {settings.STORY_TITLE}
        and a desciption of {settings.STORY_DESCRIPTION} Focus on creating complex characters
        with detailed backstories, motivations, and personalities. Ensure the characters are
        consistent with the fantasy world and the story's plot. Pay attention to crafting characters
        that are relatable and engaging for the reader. write 4000 tokens.
        """


class WorldGPT:
    def __init__(self):
        self.client = OpenAI(api_key=settings.API_KEY)

    def generate_world(self) -> str:
        response = self.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_template(),
                }
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def prompt_template(self) -> str:
        return f"""
        Generate the world for the high fantasy story with title {settings.STORY_TITLE}
        and a desciption of {settings.STORY_DESCRIPTION} Focus on creating a detailed fantasy world
        with unique elements, landscapes, and creatures. Ensure the world is consistent with the story's plot
        and the characters' backstories. Pay attention to crafting a world that is immersive and engaging for the reader.
        write 4000 tokens.
        """
