from tqdm import tqdm
import openai
import time
from openai.error import RateLimitError, ServiceUnavailableError

SKIP_CHARACTERS = ['\n', '\t', ' ']

class ApiModel():

    def __init__(self,
                 model_version,
                 n_logprobs=100):

        super()
        self.n_logprob = n_logprobs


        self.is_gpt2 = (model_version.startswith('gpt2') or model_version.startswith('distilgpt2'))
        self.is_gptj = model_version.startswith('EleutherAI/gpt-j')
        self.is_bert = model_version.startswith('bert')
        self.is_gptneo = model_version.startswith('EleutherAI/gpt-neo')
        self.is_gpt3 = model_version.startswith('gpt3')
        assert (self.is_gpt3)

        self.engine_name = model_version.replace('gpt3/', '')

        with open('openai_keys.txt') as fp:
            organization_key, api_key = fp.readlines()
        openai.organization = organization_key.replace('\n','')
        openai.api_key = api_key


    def intervention_experiment(self, interventions, multitoken=False):
        word2intervention_results = {}
        for idx, intervention in enumerate(tqdm(interventions, desc='performing interventions')):
            base_string = intervention.base_string.strip()
            alt_string = intervention.alt_string.strip()
            output_base = self.get_output(base_string)
            output_alt = self.get_output(alt_string)
            base_logprobs = output_base.choices[0].logprobs.top_logprobs
            alt_logprobs = output_alt.choices[0].logprobs.top_logprobs
            assert len(alt_logprobs[0]) == self.n_logprob
            word2intervention_results[idx] = base_logprobs, alt_logprobs

            time.sleep(0.1)

        return word2intervention_results

    def get_output(self, prompt):

        while 1:
            try:
                output = openai.Completion.create(
                    model=self.engine_name,
                    prompt=prompt,
                    logprobs=self.n_logprob,
                    max_tokens=1,
                    temperature=0,
                    top_p=0,
                )
                break
            except RateLimitError:
                print('RateLimitError, waiting to retry...')
                time.sleep(15)
            except ServiceUnavailableError:
                print('ServiceUnavailableError, waiting to retry...')
                time.sleep(10)

        return output

