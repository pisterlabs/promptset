from django.conf import settings
from django.core.management.base import BaseCommand
from openai import OpenAI

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
)


class Command(BaseCommand):
    help = "Connects with ChatGPT API and prints the response based on a given prompt"

    def handle(self, *args, **options):
        chat_log = []
        while True:
            prompt = input("User: ")
            if prompt.lower() == "exit":
                break

            chat_log.append({"role": "user", "content": prompt})
            response = self.get_chatgpt_response(prompt)
            assistant_response = response.choices[0].message.content
            print(assistant_response)
            chat_log.append({"role": "assistant", "content": assistant_response})

        print("\n")
        print("Chat log:", chat_log)

    def get_chatgpt_response(self, prompt: str):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": "Your role is to create concise, engaging, and informative Facebook posts for Epic Resume, a career consulting agency. Focus on highlighting the importance of standout resumes and LinkedIn profiles, sharing client success stories, and offering career tips. Maintain a professional yet approachable tone, encourage page interactions, and include calls-to-action for likes and follows. Ensure content respects Facebook's standards and is tailored for a diverse audience, with suggestions for enhancing posts visually. Analyze post engagement for continuous improvement in content strategy.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response
