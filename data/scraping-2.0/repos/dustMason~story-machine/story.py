import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

question_prompt = (
    "Stop your output after the question.\n"
    "The question should have 3 choices, and be rendered in the following format:\n"
    "> This is an example question?\n\n"
    "1. Example choice 1\n"
    "2. Example choice 2\n"
    "3. Example choice 3\n"
)

author = "Cressida Cowell"

# from https://en.wikipedia.org/wiki/The_Writer%27s_Journey:_Mythic_Structure_for_Writers
chapters = {
    "Ordinary world": "the hero is seen in their everyday life",
    "Call to adventure": "the initiating incident of the story happens to the hero",
    "Refusal of the call": "the hero experiences some hesitation to answer the call",
    "Meeting with the mentor": "the hero gains the supplies, knowledge, and confidence needed to commence the adventure",
    "Crossing the first threshold": "the hero commits wholeheartedly to the adventure",
    "Tests, allies, and enemies": "the hero explores the special world, faces trial, and makes friends and enemies",
    "Approach to the inmost cave": "the hero nears the center of the story and the special world",
    "The ordeal": "the hero faces the greatest challenge yet and experiences death and rebirth",
    "Reward": "the hero experiences the consequences of surviving death",
    "The road back": "the hero returns to the ordinary world or continues to an ultimate destination",
    "The resurrection": "the hero experiences a final moment of death and rebirth so they are pure when they reenter the ordinary world",
    "Return with the elixir": "the hero returns with something to improve the ordinary world",
}
chapter_titles = [t for t in chapters]

chunk_size = 400  # words


class Story:
    def __init__(self, subject: str):
        self.subject = subject
        self.synopsis = []
        self.story_parts = []
        self.choices = []
        self.question = ""
        self.chapter = 0

    async def generate(self, choice: int = -1):
        prompt = ""
        if len(self.synopsis) > 0:
            prompt += self._format_synopsis()
        else:
            prompt = self._seed_prompt()
        if choice and choice != -1:
            prompt += (
                f"\n\nAt the end of the last chapter I was asked \"{self.question}\". I selected \"{self.choices[choice - 1]}\".\n\n"
                f"Continue with the next chapter of the book, entitled \"{chapter_titles[self.chapter]}\", "
                f"in which {chapters[chapter_titles[self.chapter]]}.\n\n"
            )
            if self.chapter == len(chapters) - 1:
                prompt += f"Write the final chapter of the book, entitled {chapter_titles[self.chapter]}.\n"
            else:
                prompt += (
                              f"Write {chunk_size} words followed by a multiple-choice question that lets me decide what happens next.\n"
                          ) + question_prompt

        # print("----> start debug\n", prompt, "----> end debug\n")
        completion = ""
        async for p in self._get_completion(prompt):
            completion += p
            yield p

        if self.chapter == len(chapters) - 1:
            self.story_parts.append(completion)
        else:
            lines = completion.split("\n")
            self.story_parts.append("\n".join(lines[:-5]))
            self.question = lines[-5]
            self.choices = lines[-3:]
            self.chapter += 1
        await self._generate_next_synopsis_item()

    async def _generate_next_synopsis_item(self):
        prompt = ""
        if len(self.synopsis) > 0:
            prompt += self._format_synopsis()
        prompt += "\n\nAdd one bullet point summarizing the following. Do not extend the story to include new details.\n"
        prompt += self.story_parts[-1]
        res = ""
        async for p in self._get_completion(prompt):
            res += p
        self.synopsis.append(res)

    async def _get_completion(self, prompt: str):
        async for part in await openai.Completion.acreate(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.55,
                max_tokens=chunk_size * 2,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stream=True,
        ):
            yield part.choices[0].text

    def _format_synopsis(self):
        return (
                   f"The following bullet points summarize what's happened so far in a book written by {author} "
                   f"about {self.subject}\n"
               ) + "\n".join(self.synopsis)

    def _seed_prompt(self):
        return (
                   f"This is a book about {self.subject}. It's written in the style of {author}.\n"
                   "It's written in choose-your-own-adventure format.\n"
                   f"In the first {chunk_size} words, write the chapter entitled {chapter_titles[0]}, in which {chapters[chapter_titles[0]]}.\n"
                   "Then write a multiple-choice question that lets me decide what happens next.\n"
               ) + question_prompt
