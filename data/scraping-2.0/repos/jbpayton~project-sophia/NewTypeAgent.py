from openai import OpenAI
import util


class NewTypeAgent:
    def __init__(self, name):
        self.name = name
        # load profile
        self.profile = util.load_profile(name)

        self.MONOLOGUE_PROMPT = "Before everything you say, include internal monologue in curly brackets."

        self.EMOTION_PROMPT = "Express your emotions by leading a sentence with parenthesis with your emotional " \
                              "state. Valid emotional states are as follows: Default, Angry, Cheerful, Excited, " \
                              "Friendly, Hopeful, Sad, Shouting, Terrified, Unfriendly, Whispering."

        self.system_prompt = self.profile['personality'] + self.MONOLOGUE_PROMPT + self.EMOTION_PROMPT
        self.client = OpenAI()

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def send(self, message):
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=self.messages,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        role = response.choices[0].message.role
        content = response.choices[0].message.content
        self.messages.append({"role": role, "content": content})
        # example content, parse content into internal monologue, emotion, and the rest:
        # '{Curiosity piqued, time to clarify} (Friendly) FIRE as in Finance term, or literal fire?'
        monologue = content.split("}")[0][1:]
        emotion = content.split(")")[0].split("(")[1]
        response = content.split(")")[1]
        # trim whitespace from response
        response = response.strip()

        return response, emotion, monologue


if __name__ == "__main__":
    util.load_secrets()
    agent = NewTypeAgent("Sophia")
    while True:
        # prompt user for input
        message = input("Enter a message: ")
        # send message to agent
        response, emotion, monologue = agent.send(message)
        # print response
        print(response)


