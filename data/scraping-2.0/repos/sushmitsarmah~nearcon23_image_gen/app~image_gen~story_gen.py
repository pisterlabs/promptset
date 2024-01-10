import os
from dotenv import load_dotenv
from openai import OpenAI

from app.image_gen.ai_prompt_templates import comic_panel_prompt_2x2
from app.image_gen.utils import clean_panel_prompt

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY", "")

client = OpenAI(
    api_key=OPENAI_KEY,
)

panel_format = {
    "2x2": {
        "count": 4,
        "prompt": comic_panel_prompt_2x2
    }
}

class AIStory:

    def get_stories(self, prompt, count=10):
        prompt = """
            create {} different stories about: {}
        """.format(count, prompt)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=3000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        stories = [
            x for x in chat_completion.choices[0].message.content.split("\n") if x != ""]
        return stories

    def create_comic_panels(self, story: str, fmt="2x2"):
        template = panel_format[fmt]["prompt"]
        panel_count = panel_format[fmt]["count"]
        prompt = template.format(panel_count, story)

        story_panels = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=3000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        panels = [
            x for x in clean_panel_prompt(story_panels.choices[0].message.content).split("\n")
        ]
        exclude = [x for x in panels if x == "" or "Prompt" in x or "Panel" in x]
        panels = [x for x in panels if x not in exclude]
        return panels
