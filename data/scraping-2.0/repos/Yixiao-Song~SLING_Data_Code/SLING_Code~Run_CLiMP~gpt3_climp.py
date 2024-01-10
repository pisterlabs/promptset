import math
import numpy as np
import openai


# functions for eval
def load_climp_dataset(prefix, filename, skip_header=True, extract=lambda x: x[-2]):
    with open(f'CLiMP-data/{filename}', 'r') as f:
        dataset = f.readlines()

    if skip_header:
        dataset = dataset[1:]

    dataset = [x.split(',') for x in dataset]

    # confirm all elements of dataset have same length
    assert all([len(x) == len(dataset[0]) for x in dataset])
    assert len(dataset) % 2 == 0

    good_sent = [extract(dataset[i]) for i in range(0, len(dataset), 2)]
    bad_sent  = [extract(dataset[i + 1]) for i in range(0, len(dataset), 2)]

    return good_sent, bad_sent

def get_response(prompt: str, max_tokens = 150, temperature = 0.7, \
                 top_p = 1, n = 1, logprobs = 1, stop = None, echo = True):
    response = openai.Completion.create(engine = "davinci", 
                                        prompt = prompt, 
                                        max_tokens = max_tokens,
                                        temperature = temperature,
                                        top_p = top_p,
                                        n = n,
                                        logprobs = logprobs, 
                                        stop = stop,
                                        echo = echo)
    return response

def perplexity(log_probs):
    N = len(log_probs)
#     print(f"Sentence length is {N}.")
    return math.exp((-1/N) * np.sum(log_probs))

def evaluate_response(response, max_tokens):
    response_dict = dict(response['choices'][0])
    text = response_dict['text']
    
    log_probs = response_dict['logprobs']['token_logprobs'][1:]
    ppl_prompt = perplexity(log_probs)

    return {
        'prompt_ppl': ppl_prompt,
        'text': text
    }


"""
Eval
When evaluating the minimal pairs, the default setting of 
load_climp_dataset can be used. However, 'ba_construction_1000.csv'
needs to be run separately using the parameters set as below
because of the file format.
"""

# climp = ['anaphor_agreement_gender_1000.csv', 'binding_gender_1000.csv',
#          'classifier_1000.csv', 'classifier_adj_1000.csv',
#          'classifier_clause_1000.csv', 'coverb_instrument_1000.csv',
#          'coverb_with_1000.csv', 'filler_gap_dependency_1000.csv',
#          'head_final_clause_1000.csv', 'passive_formal_1000.csv',
#          'verb_complement_direction_1000.csv', 'verb_complement_duration_1000.csv',
#          'verb_complement_frequency_1000.csv', 'verb_complement_res_adj_1000.csv',
#          'verb_complement_res_verb_1000.csv']

climp = ['ba_construction_1000.csv']

with open("outputs/gpt3_results.txt", 'a+') as file:
    for paradigm in climp:
        file.write(f"{paradigm}\n")
        # for ba_construction_1000.csv
        good_sent, bad_sent = load_climp_dataset('CLiMP-data', paradigm, skip_header=False, extract=lambda x: x[-1])
        # for other paradigms
        # good_sent, bad_sent = load_climp_dataset('CLiMP-data', paradigm)
        
        correct, incorrect = 0, 0
        for good, bad in zip(good_sent, bad_sent):
            response_good = get_response(good, max_tokens=0)
            response_bad = get_response(bad, max_tokens=0)

            good_ppl = evaluate_response(response_good, max_tokens=0)['prompt_ppl']
            bad_ppl = evaluate_response(response_bad, max_tokens=0)['prompt_ppl']

            if good_ppl < bad_ppl:
                correct += 1
            else:
                incorrect += 1

        assert correct + incorrect == 1000

        print(f"\t{paradigm}\t{correct/1000:.4f}")
        file.write(f"\t{paradigm}\t{correct/1000:.4f}\n")
