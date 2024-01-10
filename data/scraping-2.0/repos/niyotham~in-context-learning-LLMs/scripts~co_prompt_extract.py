#@title Create the prompt (Run this cell to execute required code) {display-mode: "form"}
import cohere as co

class cohereExtractor():
    def __init__(self, examples,co):
        self.examples = examples
        self.co = co

    def make_prompt(self, example):
        examples = self.examples + [example]

        return ("".join([str(i) for i in examples]))

    def extract(self, example):
        extraction = self.co.generate(
            model='large',
            prompt=self.make_prompt(example),
            max_tokens=50,
            temperature=0.25,
            stop_sequences=["----"])

        return(extraction.generations[0].text[:-1])
