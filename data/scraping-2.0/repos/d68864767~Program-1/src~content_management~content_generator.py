import json
import openai
from src.api.openai_connector import OpenAIConnector
from src.utils.utils import load_json_config
from src.utils.logger import Logger

class ContentGenerator:
    def __init__(self):
        self.logger = Logger().get_logger()
        self.settings = load_json_config('config/project_settings.json')
        self.openai_connector = OpenAIConnector()
        self.content_management_settings = self.settings['content_management_settings']
        self.language_model_settings = self.settings['language_model_settings']
        self.logger.info("Content Generator initialized with settings: {}".format(self.content_management_settings))

    def generate_content(self, prompt, content_type=None, personalized_data=None):
        """
        Generate content based on a prompt and optional parameters.

        :param prompt: The input text prompt for content generation.
        :param content_type: The type of content to generate (e.g., blog post, product description).
        :param personalized_data: Optional data for content personalization.
        :return: Generated content as a string.
        """
        if content_type and content_type not in self.content_management_settings['content_types']:
            self.logger.error("Invalid content type: {}. Valid types are: {}".format(
                content_type, self.content_management_settings['content_types']))
            return None

        content_type = content_type or self.content_management_settings['default_content_type']
        self.logger.info("Generating content for type: {}".format(content_type))

        # Personalize content if required
        if self.content_management_settings['personalization'] and personalized_data:
            prompt = self.personalize_prompt(prompt, personalized_data)

        # Generate content using OpenAI API
        try:
            response = self.openai_connector.complete(
                prompt=prompt,
                model_name=self.language_model_settings['model_name'],
                temperature=self.language_model_settings['temperature'],
                max_tokens=self.language_model_settings['max_tokens'],
                top_p=self.language_model_settings['top_p'],
                frequency_penalty=self.language_model_settings['frequency_penalty'],
                presence_penalty=self.language_model_settings['presence_penalty']
            )
            generated_content = response.get('choices')[0].get('text').strip()
            self.logger.info("Content generated successfully.")
            return generated_content
        except Exception as e:
            self.logger.error("Error in generating content: {}".format(e))
            return None

    def personalize_prompt(self, prompt, personalized_data):
        """
        Personalize the prompt with the given data.

        :param prompt: The original prompt text.
        :param personalized_data: Data to personalize the content.
        :return: A personalized prompt.
        """
        # Implement personalization logic based on personalized_data
        # This is a placeholder for actual personalization logic
        personalized_prompt = prompt + "\n\n" + json.dumps(personalized_data)
        return personalized_prompt

# Example usage
if __name__ == "__main__":
    content_generator = ContentGenerator()
    sample_prompt = "Write a blog post about the importance of AI in modern content generation:"
    generated_blog_post = content_generator.generate_content(sample_prompt)
    if generated_blog_post:
        print(generated_blog_post)
