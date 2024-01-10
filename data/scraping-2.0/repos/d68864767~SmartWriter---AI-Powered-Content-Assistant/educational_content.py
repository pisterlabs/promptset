```python
from openai_api import OpenAI_API
from database import save_content

class EducationalContentGenerator:
    def __init__(self):
        self.openai_api = OpenAI_API()

    def generate_educational_content(self, topic, user_id, max_tokens=500):
        """
        Function to generate educational content on a given topic.
        """
        # Construct the prompt
        prompt = f"Explain the topic '{topic}' in a simple and understandable way:"

        # Generate the educational content
        educational_content = self.openai_api.generate_text(prompt, max_tokens)

        # Save the educational content to the database
        save_content(educational_content, user_id)

        return educational_content
```
