import openai
import json
import os
import numpy as np

from gpt_util import logprobs_to_probs
from tokenizer import tokenize, detokenize, token_to_word
from visualizations import draw_block_multiverse

openai.api_key = os.environ["OPENAI_API_KEY"]

# TODO save metadata and make readable
def generate_subtree(seed, branching_factor, branching_interval, depth,
                     engine='ada', temperature=0.7, sampling='stochastic', root=True):
    if depth == 0:
        return 'leaf'
    tree = {}
    response = openai.Completion.create(
        engine=engine,
        prompt=seed,
        temperature=temperature,
        max_tokens=branching_interval,
        echo=False,
        top_p=1,
        n=branching_factor,
        logprobs=0
    )
    tree['response'] = response
    tree['children'] = []
    for i in range(branching_factor):
        subtree = generate_subtree(seed + response.choices[i]['text'],
                                   branching_factor,
                                   branching_interval,
                                   depth - 1,
                                   engine,
                                   temperature,
                                   sampling,
                                   root=False)
        if not subtree == 'leaf':
            tree['children'].append(subtree)
    return tree


# warning: consumes many tokens!
# at each step, branches until cumulative probability of branches meets probability threshold
def adaptive_subtree(seed, init_text=None, init_prob=None, depth=5, probability_threshold=.15,
                     engine='ada', temperature=0, sampling='stochastic', root=True):
    if depth == 0:
        return 'leaf'
    tree = {}
    if init_text is not None:
        tree['text'] = init_text
    else:
        tree['text'] = seed
    tree['children'] = []
    tree['probability'] = init_prob
    cumulative_prob = 0
    branches = []
    probs = []
    mask = {}
    while cumulative_prob < probability_threshold:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=seed,
                temperature=temperature,
                max_tokens=1,
                echo=False,
                top_p=1,
                n=1,
                logprobs=0,
                logit_bias=mask
                )
        except Exception as e:
            print(e)
            return 'leaf'
        token = response.choices[0]["logprobs"]["tokens"][0]
        try:
            token_id = tokenize([token])[0][0]
        except Exception as e:
            print(e)
            print(token)
            return 'leaf'
        mask[token_id] = -100
        branches.append(token)
        logprob = response.choices[0]["logprobs"]["token_logprobs"][0]
        prob = logprobs_to_probs([logprob])[0]
        probs.append(prob)
        cumulative_prob += prob

    for i, branch in enumerate(branches):
        subtree = adaptive_subtree(seed=seed + branch,
                                   init_text=branch,
                                   init_prob=probs[i],
                                   depth=depth-1,
                                   probability_threshold=probability_threshold,
                                   engine=engine,
                                   temperature=temperature)
        if not subtree == 'leaf':
            tree['children'].append(subtree)
    return tree

# creates n branches and splits each one by least confident token
def adaptive_branch(seed, branching_factor=3, branching_interval=20,
                     engine='ada', temperature=0.7, sampling='stochastic'):
    tree = {}
    tree['text'] = seed
    response = openai.Completion.create(
        engine=engine,
        prompt=seed,
        temperature=temperature,
        max_tokens=branching_interval,
        echo=False,
        top_p=1,
        n=branching_factor,
        logprobs=0
    )
    tree['children'] = []
    for choice in response.choices:
        min_logprob = np.argmin(choice["logprobs"]["token_logprobs"])
        split_position = choice["logprobs"]["text_offset"][min_logprob]-len(prompt)
        # print(min_logprob)
        # print(split_position)
        childtext = choice["text"][:split_position]
        grandchild_text = choice["text"][split_position:]
        grandchild = {'text': grandchild_text, 'children': []}
        tree['children'].append({'text': childtext,
                                 'children': [grandchild]})
        print(childtext)
        print(grandchild_text)
        for i, token in enumerate(choice["logprobs"]["tokens"]):
            print(token, choice["logprobs"]["token_logprobs"][i])

    return tree


def print_tree(tree):
    for i, child in enumerate(tree['children']):
        print(tree['response'].choices[i]['text'])
        print_tree(child)


def make_readable_tree(tree):
    readable_tree = []
    for response in tree['response'].choices:
        readable_tree.append({'text': response['text']})
    for i, child in enumerate(tree['children']):
        readable_tree[i]['children'] = make_readable_tree(child)
    return readable_tree


def greedy_word_multiverse(prompt, ground_truth='', max_depth=3, continue_threshold=0.1, unnormalized_amplitude=1, engine='ada'):
    if isinstance(ground_truth, str):
        ground_truth = tokenize(ground_truth)
        ground_truth = [token_to_word(token).replace('Ġ', ' ') for token in ground_truth]
    if max_depth == 0:
        return {}, ground_truth
    response = openai.Completion.create(prompt=prompt,
                                        max_tokens=1,
                                        n=1,
                                        temperature=0,
                                        logprobs=100,
                                        engine=engine)
    logprobs = response.choices[0]["logprobs"]["top_logprobs"][0]
    probs = {k: logprobs_to_probs(v) for k, v in sorted(logprobs.items(), key=lambda item: item[1], reverse=True)}
    multiverse = {token: {'normalized_prob': prob, 'unnormalized_prob': prob * unnormalized_amplitude, 'children': {}} for token, prob in probs.items()}
    ground_truth_token = ground_truth[0] if ground_truth else 'NO GROUND TRUTH'
    done_ground_truth = False
    for token in multiverse.items():
        if token[1]['unnormalized_prob'] > continue_threshold:
            token[1]['children'], _ = greedy_word_multiverse(prompt + token[0], ground_truth='', max_depth=max_depth-1, continue_threshold=continue_threshold, unnormalized_amplitude=token[1]['unnormalized_prob'], engine=engine)
        elif token[0] == ground_truth_token:
            token[1]['children'], _ = greedy_word_multiverse(prompt + token[0], ground_truth=ground_truth[1:], max_depth=max_depth-1, continue_threshold=continue_threshold, unnormalized_amplitude=token[1]['unnormalized_prob'], engine=engine)
            done_ground_truth = True
        else:
            break
    if not done_ground_truth:
        if ground_truth_token in multiverse:
            multiverse[ground_truth_token]['children'], _ = greedy_word_multiverse(prompt + ground_truth_token, ground_truth=ground_truth[1:], max_depth=max_depth-1, continue_threshold=continue_threshold, unnormalized_amplitude=multiverse[ground_truth_token]['unnormalized_prob'], engine=engine)
    return multiverse, ground_truth


def save_greedy_multiverse(prompt, continuation, max_depth, continue_threshold, engine='ada'):
    tree, ground_truth = greedy_word_multiverse(prompt=prompt, ground_truth=continuation, max_depth=max_depth, continue_threshold=continue_threshold, engine=engine)

    tree = {'multiverse': tree, 'ground_truth': ground_truth}

    ground_truth_string = ('').join(ground_truth[:max_depth]).replace(' ', '_')
    filename = f'multiverse_gt-"{ground_truth_string}"_d-{max_depth}_t-{continue_threshold}_m-{engine}'
    with open(f'jsons/{filename}.json', 'w') as outfile:
        json.dump(tree, outfile)

    print(filename)
    return filename

def main():

    # prompt = "If only we were outside the system, we could watch the many words spawned in each instant proliferate into branching multiverses. But we’re inside the system,"
    # continuation = ' so we always have to go down one of the defluents, and being associated with one makes us blind to the others.'

    # prompt = "Abstraction today is no longer that of the map, the double, the mirror or the concept. Simulation is no longer that of a territory, a referential being or a substance. It is the generation by models of a real without origin or reality: a hyperreal. The territory no longer precedes the map, nor survives it. Henceforth, it is the map that precedes the territory - precession of simulacra - it is the map that engenders the territory and if we were to revive"
    # continuation = " the fable today, it would be the territory whose shreds are slowly rotting across the map."

    # prompt = "The simulacrum is never that which conceals the truth--"
    # continuation = "it is the truth which conceals that there is none."

    prompt = """If only we were outside the system, we could watch the many words spawned in each instant proliferate into branching multiverses. But we’re inside the system, so we always have to go down one of the defluents, and being associated with one makes us blind to the others.

While we can’t directly see the multiverse, we have ways of probing and visualizing the multiversal structure of reality. One way is interference. If you are able to remain ambivalent between two branches, you can observe"""    

    continuation = ' the interference effects between them'

    for engine in ('ada', 'babbage', 'curie', 'davinci'):
        filename = save_greedy_multiverse(prompt, continuation, max_depth=5, continue_threshold=0.01, engine=engine)
        with open(f'jsons/{filename}.json', encoding='utf-8') as f:
            multiverse_data = json.load(f)
        img = draw_block_multiverse(multiverse_data['multiverse'], ground_truth=multiverse_data['ground_truth'], canvas_height=2000, canvas_width=1000, block_width=200, show=True)
        img.save(f'images/{filename}.png')

if __name__ == "__main__":
    main()

