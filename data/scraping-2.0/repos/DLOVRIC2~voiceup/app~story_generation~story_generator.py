from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.memory import SimpleMemory

import openai
import os
from dotenv import load_dotenv


# Load the environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
env_paht = os.path.join(project_root, ".env")
load_dotenv(env_paht)


class StoryTemplates:
    """
    Class that holds all the templates required for the story generation, review and improvement.
    """

    story_template = """You are a storywriter. Given a short description, you can generate a story based on the idea in 75-100 words.
    
    Idea: {idea}"""

    review_template = """You are a story critic. Given the generated story and the initial idea, it is your job to write a feedback on how to
    imporve the story. Pay attention to things such as:
    
    1. Is the length of the story within 75-100 words?
    2. Is the story engaging?
    
    Story: {story}"""

    improve_template = """You are a storywriter. Given a generated story and a review from a critic, it is your job to improve the story.
    Make sure you set the story length to MAXIMUM 150 words.
    Story: {story}
    Review: {review}
    """


class StoryGenerator:
    def __init__(self, api_key: str = None, model: str = "gpt3.5-turbo"):

        key = os.environ.get("OPENAI_KEY", api_key)
        if not key:
            raise ValueError("OPENAI API key must be provided.")
        self.llm = OpenAI(temperature=0.9, openai_api_key=key)

    
    def generate_story(self, idea):
        """
        Method that uses llm chains to generates a story, reviews and modifies the story
        accordingly.

        Args:
            idea: Input from the user on the story idea.

        Returns:
            str - LLM generated story
        """

        # Story generation
        story_template = PromptTemplate(input_variables=["idea"], template=StoryTemplates.story_template)
        story_chain = LLMChain(llm=self.llm, prompt=story_template, output_key="story")

        # Review
        review_template = PromptTemplate(input_variables=["story"], template=StoryTemplates.review_template)
        review_chain = LLMChain(llm=self.llm, prompt=review_template, output_key="review")

        # Improve
        improve_template = PromptTemplate(input_variables=["story", "review"], template=StoryTemplates.improve_template)
        improve_chain = LLMChain(llm=self.llm, prompt=improve_template)

        final_chain = SequentialChain(
            chains=[story_chain, review_chain, improve_chain],
            input_variables=["idea"],
            verbose=True
        )

        return final_chain.run(idea)


if __name__ == "__main__":

    chatbot = StoryGenerator()

    # Testing
    print(chatbot.generate_story("Story about a hackathon where a team of engineers is using elevenlabs to develop voice applications."))
