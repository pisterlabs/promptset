import re

import openai

element_list = {"Introductions", "Travel", "Food", "Shopping", "Family", "Hobbies", "Routine", "Health", "Work",
                "Culture"}


def remove_non_digits(input_string):
    # Use a regular expression to find all non-digit characters and remove them
    result_string = re.sub(r'[^0-9]', '', input_string)
    return result_string


def find_matching_element(input_string):
    for element in element_list:
        if element in input_string:
            return element
    return None


class GPT:
    def __init__(self, init_message):
        with open("API_KEY.txt") as mytxt:
            for line in mytxt:
                self.API_KEY = line
                openai.api_key = line
                self.messages = init_message  # = [ {"role": "system", "content": "You are a intelligent assistant."} ]

    def makeCall(self, model, message):
        if message:
            self.messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.chat.completions.create(
                model=model, messages=self.messages
            )
            reply = chat.choices[0].message.content
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        else:
            Exception("Please put in a message")

    def score(self):
        self.messages.append(
            {"role": "user", "content": "Respond to this only in a single number, no extra text: Grade this "
                                        "conversation out of 100 in the user's fluency with this language. be harsh!"},
        )
        chat = openai.chat.completions.create(
            model="gpt-4", messages=self.messages
        )
        reply = chat.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return int(reply)


def hi():
    g = GPT([{"role": "system", "content": "You are a intelligent assistant."}])
    g2 = GPT([{"role": "assistant", "content": "Every time we input a sentence, you have to give me a 1 word topic (no "
                                               "extra text) for"
                                               "the current topic in the conversation that it falls under. Choose "
                                               "from the following:"
                                               "-Introductions -Travel -Food -Shopping -Family -Hobbies -Routine "
                                               "-Health -Work"
                                               "-Culture"}])
    bro = ""

    while True:
        hello = input("User: ")
        if (hello == "q"):
            print(bro)
            exit(0)
        print("ChatGPT: " + g.makeCall("gpt-4", hello))
        bro = g2.makeCall("gpt-4", hello)


# hi()
