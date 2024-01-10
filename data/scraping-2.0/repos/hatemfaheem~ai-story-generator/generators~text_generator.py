import openai

from data_models import StoryText
from processors.text_processor import TextProcessor


class TextGenerator:
    _MAX_TOKENS: int = 1024

    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor

    def generate_story_text(self, prompt: str) -> StoryText:
        """Generate story text for the given prompt

        Args:
            prompt: The prompt/sentence to generate a story about

        Returns: The AI generated story text
        """
        story_content = openai.Completion.create(
            model="text-davinci-003",
            prompt="Give me a story about " + prompt,
            max_tokens=self._MAX_TOKENS,
            temperature=0,
        )
        story_raw_text = story_content["choices"][0]["text"]
        processed_sentences = self.text_processor.process_story_text(
            story_raw_text=story_raw_text
        )
        print(f"Raw story text: {story_raw_text}")
        return StoryText(
            raw_text=story_raw_text, processed_sentences=processed_sentences
        )
