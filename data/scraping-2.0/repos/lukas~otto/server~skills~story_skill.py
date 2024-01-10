from dotenv import load_dotenv
from skills.base import Skill
import openai
from openai import ChatCompletion

load_dotenv()


class StorySkill(Skill):
    function_name = "story"
    parameter_names = ['description']

    examples = [
        [
            "Tell me a story"
            "story()"
        ],
        [
            "Tell me a story about a cat and a dog",
            "story(description=\"cat and dog\")"
        ],
        [
            "Make up a story about two best friends that fly to the moon",
            "story(description=\"two best friends that fly to the moon\")"
        ]]

    def start(self, args: dict[str, str]):
        prompt = "Tell me a story"
        if ('description' in args):
            prompt += args['description']

        self.message_function("Ok I'm thinking of a story...")
        s = self._ask_gpt_story(prompt)

    # Could also do this with llama
    def _ask_gpt_story(self, question: str):
        response = ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a story telling assistant."},
                {"role": "user", "content": question},
            ]
        )
        answer = response["choices"][0]["message"]["content"]
        self.message_function(answer)


if __name__ == '__main__':
    # for testing
    story_skill = StorySkill(print)
    story_skill.start({"description": "frog and moose"})
