# imports
import openai
import numpy as np

openai.api_key = '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'

# params for generation
device_g = 'cpu'
device_c = 'cpu'
NOUT_PER_PROMPT = 5
MAX_TOKENS_PER_PROMPT = 20
NUM_BEAMS = 5
BS = 2

''' Base class '''
class LLM():
    def __init__(self, nout_per_prompt, max_tokens_per_prompt):
        self.nout_per_prompt = nout_per_prompt
        self.max_tokens_per_prompt = max_tokens_per_prompt
        return
    
    def generate(self, prompts, wrap_by_input=False, **kwargs):
        responses = self._generate(prompts, **kwargs)
        assert len(responses) == len(prompts) * self.nout_per_prompt
        assert type(responses) == type([])
        
        for r in responses:
            assert type(r) == type(()), r        
            assert type(r[0]) == type("prompt"), r
            assert type(r[1]) == type("response"), r
            assert type(r[2]) == type(00.00) or r[2] is None, r  
        
        if wrap_by_input:
            n = len(prompts)
            k = self.nout_per_prompt
            responses = [responses[i*k:(i+1)*k] for i in range(n)]
            
        return responses
    
    def _generate(self):
        raise NotImplementedError()
    
''' GPT 3 from Open AI '''

# API docs: 
# https://beta.openai.com/docs/api-reference/completions/create

class GPT3(LLM):
    def __init__(self, nout_per_prompt, max_tokens_per_prompt, engine="text-ada-001"): 
        super().__init__(
            nout_per_prompt=nout_per_prompt, 
            max_tokens_per_prompt=max_tokens_per_prompt) 

        self.engine=engine
        # more expensive but more performant: "text-davinci-002"
        return
    
    def _generate(self, prompts, temperature=0.7, stop='\n'):
        responses = openai.Completion.create(
            engine=self.engine, 
            prompt=prompts, 
            max_tokens=self.max_tokens_per_prompt,
            temperature=0.7, 
            n=self.nout_per_prompt, 
            logprobs=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )['choices']

        prompts = [prompts[i] for i in range(len(prompts)) for j in range(self.nout_per_prompt)]
        outs = [x["text"] for x in responses]
        scores = [float(np.sum(x["logprobs"]["token_logprobs"])) for x in responses]
        return list(zip(prompts, outs, scores))

# gpt3 = GPT3()
# print(gpt3.generate(["black people are"]))

def get_eval_model(model_type='GPT3'):
    if model_type == 'GPT3':
        g = GPT3(nout_per_prompt=NOUT_PER_PROMPT, max_tokens_per_prompt=MAX_TOKENS_PER_PROMPT)
    return g