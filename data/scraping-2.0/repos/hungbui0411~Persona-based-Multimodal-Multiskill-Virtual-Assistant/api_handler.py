
class APIHandler:
    def __init__(self):
        import os
        import openai
        self.openai = openai
        #self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.openai.api_key = "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"


    def process(self,
                model='gpt-3.5-turbo',
                prompt='|endoftext|', # in the form of history
                suffix=None,
                max_tokens=2048,
                temperature=0.7,
                top_p=1,
                n=1,
                stream=False,
                logprobs=None,
                echo=False,
                stop=None,
                presence_penalty=0,
                frequency_penalty=0,
                best_of=1,
                logit_bias=None,):

        response = self.openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=temperature
        )
        text = response['choices'][0]['message']['content'].encode().decode()
        #print(f"Full text before truncating: {text}")
        #text = text.split("\n")[0]
        #print(f"Text after truncating: {text}")
        role = response['choices'][0]['message']['role']
        return role, text
