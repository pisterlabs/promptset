from dataclasses import dataclass
import openai


client = openai.Client()

@dataclass()
class StoryGenerator:
    prompt: str
    model: str


    def generate(self):
        resp = client.chat.completions.create(
            messages=[
                {
                    "role": "system", "content": self.prompt,
                },
            ], model=self.model,
            max_tokens=4000,
        )

        return resp.choices[0].message.content

        
