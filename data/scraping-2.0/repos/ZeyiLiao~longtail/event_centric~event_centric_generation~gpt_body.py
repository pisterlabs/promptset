
import openai

with open('/home/zeyi/key.txt') as f:
    key = f.read()
openai.api_key = key





class PromptWrapper:
    def __init__(self, config):
        self.config = config
    
    def prompt_generation(self,input, prefix, neg = False):

        if neg:
            prompt_str = self.create_prompt_neg(prefix, input)
        else:
            prompt_str = self.create_prompt(prefix, input)

        response = openai.Completion.create(
            prompt=prompt_str,
            **self.config.__dict__,
        )
        target = [_.text.strip() for _ in response.choices]
        
        return self.check_generation(target)


    def check_generation(self,l):
        good_generation = []
        for text in l:
            if text[-1] != '.':
                continue
            good_generation.append(text)
        return good_generation


    def create_prompt_neg(self, prefix, input):
        return f"{prefix}\n" \
               f"Original: {input}\n" \
               f"Negated: "

    def create_prompt(self, prefix, input):
        return f"{prefix}\n" \
               f"{input}"

