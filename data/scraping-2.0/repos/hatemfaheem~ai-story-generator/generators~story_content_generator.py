import openai

from data_models import StoryPageContent, StoryContent, StorySize
from generators.image_generator import ImageGenerator
from generators.text_generator import TextGenerator
from util.openai_credentials_provider import OpenAICredentialsProvider


class StoryContentGenerator:
    def __init__(
        self,
        text_generator: TextGenerator,
        image_generator: ImageGenerator,
        credentials_provider: OpenAICredentialsProvider,
    ):
        openai.organization = credentials_provider.organization
        openai.api_key = credentials_provider.api_key
        self.image_generator = image_generator
        self.text_generator = text_generator

    def generate_new_story(
        self, workdir_images: str, story_seed_prompt: str, story_size: StorySize
    ) -> StoryContent:
        """Generate a new story for the given prompt

        Args:
            workdir_images: The workdir where images should be stored
            story_seed_prompt: The title/seed of the story
            story_size: Story size configuration

        Returns: The contents of the newly generated story
        """
        story_text = self.text_generator.generate_story_text(story_seed_prompt)
        raw_text = story_text.raw_text
        processed_sentences = story_text.processed_sentences
        page_contents = []

        for i in range(len(processed_sentences)):
            image_prompt = (
                f"A painting for '{processed_sentences[i]}'. "
                f"{story_seed_prompt}."
            )
            url = self.image_generator.generate_image(
                prompt=image_prompt, story_size=story_size
            )
            image_number: str = str(i).zfill(3)
            image, image_path = self.image_generator.download_image(
                workdir=workdir_images,
                url=url,
                image_number=image_number,
            )
            story_page_content = StoryPageContent(
                sentence=processed_sentences[i],
                image=image,
                image_path=image_path,
                page_number=image_number,
            )
            page_contents.append(story_page_content)

        return StoryContent(
            story_seed=story_seed_prompt,
            raw_text=raw_text,
            page_contents=page_contents,
            story_size=story_size,
        )
