import yaml
import json
import os
import asyncio

from utils.openaicaller import openai
from utils.normalize_file import normalize_file as nf
from utils.config import bcolors
from utils.misc import clear_screen, open_explorer_here, realbcolors, printm

from classes.video import Video

desc_prompt = """Write an engaging and humorous description for the YouTube channel "[name]" that will be placed on the channel's about page. The description should aim to entice viewers to subscribe to the channel. It should be creative, captivating, and slightly longer, but still within a reasonable length. Try to keep it under 1,000 characters to ensure it remains concise and attention-grabbing. Make sure to incorporate the channel's subject, which is "[subject]" Remember that this description will not be part of the channel's videos.
Answer exclusively with the description, and no other form of greeting or salutation, like "Sure, I'll do that!" or "Here's the description:".
"""
ideas_prompt = """You will generate a list of ideas of videos about [subject]. You will suggest topics for videos that will be created and posted on YouTube.
You will output the list of ideas in a json format. The following fields will be included:
- title
- description
Here is an example of the output:
```
[
    {
        "title": "TITLE OF THE VIDEO",
        "description": "A video about something. Concept1 and concept2 are explained. Concept3 is also explained a bit. We also talk about this and this.
    },
    {
        "title": "TITLE OF THE VIDEO",
        "description": "A video about something. In this video we will create a thing. We will also see how that is possible. We will also talk about this and this.
    },
    {
        "title": "TITLE OF THE VIDEO",
        "description": "A video about something. We will create the following project, from a to z. We will see how to do this and this. We will also talk about this and this.
    },
    {
        "title": "TITLE OF THE VIDEO",
        "description": "A video about the story of how John Doe created a thing. We will see how he did it. We will also talk about this and this.
    },
]
```
You will not answer anything else in your message. Your answer will only be the json output, without any other text. This is very important. no codeblock, nothing like "Here are .....". Just the json. You will generate 10 ideas. You will never repeat yourself.
Here are the existing ideas wich you should not repeat again.
[existing ideas]
"""

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    openai.set_api_key(os.getenv("OPENAI_API_KEY"))

class Channel:
    def __init__(self) -> None:
        pass
    async def create(self):
        printm("We assume that you have a YouTube channel. If you don't, please create one before continuing.")
        input("Press enter when you are ready.")
        printm("Great, let's go!")
        self.name = input(f"First, please tell me the name of your YouTube channel. Press enter when you are done:{bcolors.BOLD}{bcolors.OKCYAN} ")
        printm(f"{bcolors.ENDC}", end="")
        self.path = os.path.join("channels", f"{self.name}")
        self.path = os.path.abspath(self.path)
        if os.path.exists(self.path):
            raise FileExistsError("Channel already exists")
        os.makedirs(self.path)
        printm(f"Great! {bcolors.BOLD}{self.name}{bcolors.ENDC} is a great name for a YouTube channel!")
        self.subject = input(f"Now, please tell me the subject of your YouTube channel. Press enter when you are done:{bcolors.BOLD}{bcolors.OKCYAN} ")
        printm(f"{bcolors.ENDC}", end="") # we use end="" to avoid a new line
        printm(f"Great! {bcolors.BOLD}{self.subject}{bcolors.ENDC} is a great subject for a YouTube channel!")
        printm("Now, I will generate a description for your channel. Please wait...")
        response = await openai.generate_response(model="gpt-3.5-turbo", messages=[{'role':'user', 'content': desc_prompt.replace("[name]", self.name).replace("[subject]", self.subject)}], max_tokens=100, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0.6, stop=["\n\n"])
        self.description  = response['choices'][0]['message']['content'] # type: ignore
        printm(f"Great! Here is the description I generated for your channel: \n**{self.description}**")
        printm(f"Now, please paste all the needed file(s) in the folder that will open. Press enter to open the folder.")
        input()
        open_explorer_here(self.path)
        with open(f"{self.path}/description.txt", "w") as f:
            f.write(self.description)
            f.close()
        self.data = {
            "name": self.name,
            "subject": self.subject,
            "description": self.description,
            "path": self.path,
        }
        await self.generate_ideas()

        with open(f"{self.path}/channel.yaml", "w") as f:
            yaml.dump(self.data, f)
            f.close()
        input(f"You can sleep now if you are tired. Press enter when you are awake and ready to continue {bcolors.BOLD}{bcolors.OKCYAN}:{bcolors.ENDC}{bcolors.BOLD}{bcolors.WARNING}){bcolors.ENDC}")
        clear_screen()
        return self.data

    async def load(self, name):
        self.name = name
        self.path = f"channels/{name}"
        if not os.path.exists(self.path):
            raise FileNotFoundError("Channel not found")
        with open(f"{self.path}/channel.yaml", "r") as f:
            self.data = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        self.ideas = []
        if os.path.exists(os.path.join(self.path, "ideas.json")):
            with open(os.path.join(self.path, "ideas.json"), "r") as f:
                self.ideas = json.load(f)
                f.close()
        self.name = self.data['name']
        self.subject = self.data['subject']
        self.description = self.data['description']
        self.path = self.data['path']
        return self.data

    async def generate_ideas(self):
        if not os.path.exists(os.path.join(self.path, "ideas.json")):
            ideas = []
        else:
            with open(os.path.join(self.path, "ideas.json"), "r") as f:
                ideas = json.load(f)
                f.close()
        response = await openai.generate_response(model="gpt-3.5-turbo", messages=[{'role':'user', 'content': ideas_prompt.replace("[subject]", self.subject).replace("[existing ideas]", "\n".join([f"- {idea['title']}" for idea in ideas]))}])
        string_new_ideas = response['choices'][0]['message']['content'] # type: ignore
        new_ideas = json.loads(string_new_ideas)
        ideas += new_ideas
        with open(os.path.join(self.path, "ideas.json"), "w") as f:
            json.dump(ideas, f, indent=4)
            f.close()
        self.ideas = ideas
        return ideas
    
    async def generate_video(self, idea):
        #get the idea object from self.ideas
        if not idea in self.ideas:
            raise ValueError("Idea not found")
        if not os.path.exists(os.path.join(self.path, "videos")):
            os.makedirs(os.path.join(self.path, "videos"))
        self.video = Video(idea, self)
        await self.video.generate()
        return self.video

    

if __name__ == "__main__":
    
    async def main():
        channl = Channel()
        await channl.create()
    
    asyncio.run(main())