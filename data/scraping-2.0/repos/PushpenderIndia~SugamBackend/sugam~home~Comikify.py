import openai

class Comikify:
    def __init__(self, topic, openai_key, language):
        self.topic = topic
        self.openai_key = openai_key
        self.prompt = f'write a script dialogue in {language} language on topic: "{self.topic}", script should be generated in such a way so that we can understand the whole topic'


    def start(self):
        openai.api_key = self.openai_key
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=self.prompt,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5,
        )

        dialogue = response.choices[0].text.strip().split('\n')
        final_dialogue = []
        for raw in dialogue:
            if raw.strip() != "":
                final_dialogue.append(raw.strip().split(':')[1].strip())

        if len(final_dialogue) < 2:
            return ["Character 1: [Dialogue missing]", "Character 2: [Dialogue missing]"]

        return final_dialogue

if __name__ == "__main__":
    # Usage example
    topic = "Artificial Intelligence"
    openai_key = "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"

    generator = Comikify(topic, openai_key, "Hindi")
    dialogue = generator.start()
    print(dialogue)

    for line in dialogue:
        print(line)
