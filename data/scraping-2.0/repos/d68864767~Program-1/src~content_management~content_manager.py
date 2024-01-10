import json
from src.utils.logger import Logger
from src.api.openai_connector import OpenAIConnector
from src.utils.utils import load_json_config

class ContentManager:
    def __init__(self):
        self.logger = Logger().get_logger()
        self.settings = load_json_config('config/project_settings.json')['content_management_settings']
        self.openai_connector = OpenAIConnector()
        self.content_types = self.settings['content_types']
        self.default_content_type = self.settings['default_content_type']
        self.personalization = self.settings['personalization']
        self.content_curation = self.settings['content_curation']
        self.copy_editing = self.settings['copy_editing']

    def generate_content(self, content_type=None, prompt="", user_data=None):
        if content_type not in self.content_types:
            content_type = self.default_content_type
            self.logger.warning(f"Invalid content type provided. Defaulting to {self.default_content_type}.")

        if self.personalization and user_data:
            prompt = self.personalize_prompt(prompt, user_data)

        content = self.openai_connector.generate_text(prompt)
        
        if self.copy_editing:
            content = self.edit_content(content)

        return content

    def personalize_prompt(self, prompt, user_data):
        # Implement personalization logic here
        # This is a placeholder for the actual personalization logic
        personalized_prompt = f"{prompt}\n\nUser data: {json.dumps(user_data)}"
        return personalized_prompt

    def edit_content(self, content):
        # Implement copy editing logic here
        # This is a placeholder for the actual copy editing logic
        edited_content = content.replace('  ', ' ')  # Example: removing extra spaces
        return edited_content

    def curate_content(self, content_list):
        if not self.content_curation:
            self.logger.info("Content curation is disabled in the settings.")
            return content_list

        # Implement content curation logic here
        # This is a placeholder for the actual content curation logic
        curated_content = sorted(content_list, key=lambda x: len(x))
        return curated_content

# Example usage
if __name__ == "__main__":
    content_manager = ContentManager()
    generated_content = content_manager.generate_content(
        content_type="blog_posts",
        prompt="Write a blog post about the importance of AI in modern content generation.",
        user_data={"name": "John Doe", "interests": ["AI", "Technology", "Content Marketing"]}
    )
    print(generated_content)
