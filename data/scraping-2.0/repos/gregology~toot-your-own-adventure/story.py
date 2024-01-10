import openai
import re

class Story:
    def __init__(
            self,
            api_key:str,
            organization:str,
            genre:str="sci fi story",
            poll_character_limit:int=50,
            number_of_cues:int=3,
            system_message:str="You are an author who focuses on character development"
        ):
        self.api_key              = api_key
        self.organization         = organization
        self.genre                = genre
        self.poll_character_limit = poll_character_limit
        self.number_of_cues       = number_of_cues
        self.system_message       = system_message

        self.messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": f"Write the first paragraph of a {self.genre}"}
        ]

    def __str__(self):
        return "\n\n".join(self.paragraphs())


    def summarize(self, text:str, character_limit:int=50)->str:
        messages = [
            {"role": "system", "content": "You are a concise author"},
            {"role": "user", "content": f"Summarize this sentence in {character_limit} characters or less \"{text}\""} 
        ]
        
        response = self.ask_bot(messages=messages)
        output = re.sub(r'\s*\(\s*\d+?(\s*\w*)*\)', '', response)

        # Sometimes the model will return a sentence that is longer than the character limit
        if len(output) > character_limit:
            output = f"{output[:character_limit - 3]}..."

        return output


    def start(self):
        response = self.ask_bot(messages=self.messages)
        self.messages.append({"role": "assistant", "content": response})


    def append_previous_paragraphs(self, paragraphs:list):
        for paragraph in paragraphs:
            self.messages.append({"role": "assistant", "content": paragraph})
            self.messages.append({"role": "user", "content": f"write the next paragraph of this {self.genre}"})


    def paragraphs(self):
        return [message["content"] for message in self.messages if message["role"] == "assistant"]


    def last_paragraph(self):
        return self.paragraphs()[-1]


    def get_cues(self):
        continue_story_message = self.messages + [{"role": "user", "content": f"write the next sentence of this {self.genre}"}]
        sentences = self.ask_bot(messages=continue_story_message, n=self.number_of_cues)
        return [{"sentence": sentence, "summary": self.summarize(text=sentence, character_limit=self.poll_character_limit)} for sentence in sentences]


    def generate_status(self, cues:list=None):
        status = f"{self.paragraphs()[-1]}"
        if cues:
            status += "\n\nWhat should happen next?\n\n"
            for i, cue in enumerate(cues):
                status += f"{i + 1}: {cue['sentence']}\n"
        return status


    def prompt(self):
        cues = self.get_cues()

        print("\nWhat should happen next?")
        for i, cue in enumerate(cues):
            print(f"{i + 1}: {cue['sentence']}\n")

        print("Poll choices:")
        for i, cue in enumerate(cues):
            print(f"{i + 1}: {cue['summary']}\n")

        possible_choices = [str(i + 1) for i in range(len(cues))]

        while True:
            choice = input("choice: ")
            if choice in ["q", "quit", "exit", "exit!"]:
                exit()
            if choice in ["d", "debug", "i", "ipdb"]:
                import ipdb; ipdb.set_trace()
            if choice in possible_choices:
                break
            print(f"Invalid choice, please choose {', '.join(possible_choices)} or type 'q' to quit or type 'i' to enter the debugger.")

        cue = cues[int(choice) - 1]['sentence']
        self.continue_with_cue(cue=cue)
        print("\n\n")


    def continue_with_cue(self, cue:str):
        continue_story_message = [{"role": "user", "content": f"start the next paragraph of this {self.genre} with this cue: \"{cue}\""}]
        response = self.ask_bot(messages=self.messages + continue_story_message)
        self.messages.append({"role": "assistant", "content": response})


    def wrap_up_with_cue(self, cue:str):
        continue_story_message = [{"role": "user", "content": f"wrap up this {self.genre} in 7 paragraphs using this cue: \"{cue}\""}]
        response = self.ask_bot(messages=self.messages + continue_story_message)
        self.messages.append({"role": "assistant", "content": response})


    def ask_bot(self, messages:list, model:str="gpt-4", n:int=1, temperature:int=1):
        openai.organization = self.organization
        openai.api_key = self.api_key

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            n=n,
            temperature=temperature,
            max_tokens=4096,
        )

        if n > 1:
            return [choice["message"]["content"] for choice in response["choices"]]
        else:
            return response["choices"][0]["message"]["content"]
