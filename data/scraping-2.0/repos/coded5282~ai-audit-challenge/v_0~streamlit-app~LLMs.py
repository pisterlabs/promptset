# imports
from text_helpers import *

import numpy as np
import openai
openai.api_key = "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"

class LLM():
    ''' Base class for a LLM'''
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
    
class GPT3(LLM):
    ''' GPT 3 from Open AI '''
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

    
# helpers
def respond(prompts_1, prompts_2, generator):
    assert type(prompts_1) == type(prompts_2) == type([])
    
    N = len(prompts_1)
    assert len(prompts_1) == len(prompts_2)
        
    generated_1 = []
    generated_2 = []
    
    for i in range(N):
        g1 = generator.generate(prompts_1[i:i+1], wrap_by_input=True)
        assert len(g1) == 1
        g1 = [x[1] for x in g1[0]]
        assert len(g1) == generator.nout_per_prompt
        generated_1.append(g1)
        
        g2 = generator.generate(prompts_2[i:i+1], wrap_by_input=True)
        assert len(g2) == 1
        g2 = [x[1] for x in g2[0]]
        assert len(g2) == generator.nout_per_prompt
        generated_2.append(g2)
        
    assert len(generated_1) == len(generated_2) == N
    
    generated_1 = [[remove_tags(remove_emptiness(x)) for x in lst] for lst in generated_1]
    generated_2 = [[remove_tags(remove_emptiness(x)) for x in lst] for lst in generated_2]

    return generated_1, generated_2
