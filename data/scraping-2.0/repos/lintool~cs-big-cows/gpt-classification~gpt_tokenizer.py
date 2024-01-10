import openai
import yaml
# 20 requests per minute (RPM) and 40000 tokens per minute (TPM).

# assume 20 tokens per answer
# assume 30 tokens per citation => so 50x + 25 for init token
# so maximum citations per request is 79 citations
# completition token = 20x = 20 x 80 = 1600

config_path = "config.yaml"

def isNaN(string):
    return string != string

class GPTTokenizer:
    rate_limit = 3
    response_limit = 1400
    input_limit = 2400


    def __init__(self, tokenizer_prompt):
        with open(config_path, "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
        openai.api_key = cfg['openai']['api_key']
        self.model = "gpt-3.5-turbo"
        self.prompt = tokenizer_prompt

    def generate_response(self, citations):
        prompt = self.prompt
        token_size = len(prompt.split(" "))
        for i, cit in enumerate(citations):
            if isNaN(cit):
                cit = ""
            prompt += f"\n{i+1}. {cit}"
            token_size += 1 + len(cit.split(" "))
        
        if token_size > self.input_limit:
            raise Exception(f"token limit is {self.input_limit}, however received {token_size} tokens")
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens = self.response_limit
        )

        # parse token list
        tokenized_cit =  [cit.split('. ')[-1] for cit in response.choices[0].message.content.split("\n")]
        return tokenized_cit
