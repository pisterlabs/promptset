from dataclasses import dataclass
from transformers import AutoModel,AutoConfig,AutoModelForCausalLM,AutoTokenizer
import torch
import math
import os
import openai

with open('/home/zeyi/key.txt') as f:
    key = f.read()
openai.api_key = key


class GPTppl():
    def __init__(self,device):

        self.model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.device = torch.device(device)
        self.model.to(self.device)

    def calculate_ppl(self,composed_rules):
        composed_rules = list(set(composed_rules))
        features = [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(composed_rule))) for composed_rule in composed_rules]
        ppl_all = []

        self.model.eval()

        with torch.no_grad():
            for index,feature in enumerate(features):
                feature = feature.to(self.device)
                loss = self.model(
                    feature,
                    labels = feature,
                    return_dict = True
                ).loss
                ppl_all.append(math.exp(loss.item()))

        return ppl_all




def filter_by_format(input,outputs,constraints, no_filter = False):

    generations = []

    for output in outputs:
    #   filter those not follow the mask pattern
        
        if output[-1] == '.':
            output = output[:-1]
        
        if output[:len(input)] == input:
            generation = output.replace(input,'').strip()

            if no_filter:
                generations.append(generation)
            else:
                clause_states = []

                for constraint in constraints:

                    #  filter those not follow the constraints
                    clause_satisified = False

                    for concept in constraint:
                        if concept in generation or (concept.capitalize()) in generation:
                            clause_satisified = True
                            break

                    clause_states.append(clause_satisified)

                if all(clause_states):
                    generations.append(generation)
                else:
                    generation = 'Filtered'
                    generations.append(generation)

        else:
            
            generation = 'Head_dont_match'
            generations.append(generation)



    return generations





@dataclass
class PromptConfig:
    engine: str = "text-davinci-002"
    max_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 1
    logprobs: int = 0
    n: int = 5
    echo: bool = False




class PromptWrapper:
    def __init__(self, prefix: str, no_filter = False):
        self.prefix = prefix
        self.negation_config = PromptConfig()
        self.ppl = GPTppl('cuda')
        self.no_filter = no_filter


    def prompt_generation(self,input, inflection_constraint, lemma_constraint):


        prompt_str = self.create_prompt(input,lemma_constraint)
        
        response = openai.Completion.create(
            prompt=prompt_str,
            **self.negation_config.__dict__,
        )
        
        target = self.filter_generations(response.choices,input,inflection_constraint)
        return target


    def filter_generations(self,explanations,input,constraints):
        # Extract string explanations

        _explanations = []
        for explanation in explanations:
            text = explanation.text
            if text[-1] != '.':
                text += '.'
            _explanations.append(text.strip())

        _explanations = list(set(_explanations))
        
        generations = filter_by_format(input,_explanations,constraints,no_filter = self.no_filter)
        
        error = ['Head_dont_match','Filtered']
        generations = [_ for _ in generations if _ not in error]
        
        if self.no_filter:
            if len(generations) != 0:
                back_generations = [f'{input} {_}.' for _ in generations]
                sorted_index = sorted(range(len(back_generations)), key= lambda i :self.ppl.calculate_ppl(back_generations)[i])
                generations = [exp for (i,exp) in enumerate(generations) if i in sorted_index]
            else:
                return []

        return generations


    def create_prompt(self, input: str, constraints: str):
        return f"{self.prefix}\n" \
               f"Constraints: {constraints} ; Input: {input} ; Output:\n"\
               f"Output:"
