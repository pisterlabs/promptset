import openai
import replicate

import itertools
import os
import time
import pickle
import random

from utilities import *
from secret_keys import retrieve_keys
from db import SQLiteCache

from func_timeout import func_timeout, FunctionTimedOut

keys=retrieve_keys()
random.shuffle(keys)

_sampling_seed = 0
def set_sampling_seed(seed):
    global _sampling_seed
    _sampling_seed = seed
def randomly_sample_with_seed(choices, n):
    global _sampling_seed
    # save the old seed
    old_seed = random.getstate()
    # set the new seed
    random.seed(_sampling_seed)
    # sample
    sampled = random.sample(choices, n)
    # restore the old seed
    random.setstate(old_seed)
    return sampled



def sample_completions(prompt, n, temperature=1, stop="\n", max_tokens=128, engine="code-davinci-002"):
    batch_size = 17 if "gpt-4" in engine else 128
    sampled = 0
    while sampled < n:
        this_batch_size = min(batch_size, n-sampled)
        _sample_completions(prompt, this_batch_size+sampled,
                            temperature=temperature, stop=stop, max_tokens=max_tokens, engine=engine)
        sampled += this_batch_size

    return _sample_completions(prompt, n,
                               temperature=temperature, stop=stop, max_tokens=max_tokens, engine=engine, randomly_subsample=True)


completions_database = SQLiteCache("completions.db")
def _sample_completions(prompt, n, temperature=1, stop="\n", max_tokens=128, engine="code-davinci-002", randomly_subsample=False):
    global keys
    
    existing_completions = completions_database.n_entries(prompt, engine, temperature, max_tokens, stop)
    if existing_completions >= n:

        if randomly_subsample:
            return_value = completions_database.lookup(prompt, engine, temperature, max_tokens, stop, n=existing_completions)
            return_value = random.sample(return_value, n)
            assert len(return_value) == n
            return return_value


        return_value = completions_database.lookup(prompt, engine, temperature, max_tokens, stop, n=n)
        assert len(return_value) == n
        return return_value

    assert not randomly_subsample, f"assumes that you already have all the samples. you have {existing_completions} but want {n}"

    if existing_completions > 0:
        print(f"extending cached completions: {existing_completions}->{n}")
    else:
        print("new cached completion")

    if engine == "llama":
        completions = []

        for i in range(n-existing_completions):
            producer = replicate.run(
                "replicate/llama-2-70b:14ce4448d5e7e9ed0c37745ac46eca157aab09061f0c179ac2b323b5de56552b",
                input={"prompt": prompt, "max_length": max_tokens, "temperature": max(0.01,temperature)}
            )

            new_completion = ""
            for produced in producer:
                new_completion += produced
                if isinstance(stop, str) and stop in produced:
                    new_completion = new_completion[:new_completion.index(stop)]
                    break
                if isinstance(stop, list) and any([s in produced for s in stop]):
                    # again we need to take the prefix for the first stop token
                    # that appears
                    for s in stop:
                        if s in produced:
                            new_completion = new_completion[:new_completion.index(s)]
                            break
                    break
                
                if len(new_completion) >= max_tokens:
                    break

            completions.append((new_completion, 0, 0))

        result = completions
        

        
    else:
        backoff=1
        print("Querying OpenAI...")
        
        while True:
            
            try:
                openai.api_key=keys[0]
                keys = keys[1:] + [keys[0]]
                if engine in ["gpt-3.5-turbo", "gpt-4"]:
                    hypotheses = func_timeout(1000, openai.ChatCompletion.create, 
                    kwargs={"model": engine, 
                        "messages":[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature":temperature,
                        "n":n-existing_completions,
                        "stop":stop,
                        "max_tokens":max_tokens
                    })
                    result = [ (h["message"]["content"],
                                0, 0)
                            for h in hypotheses["choices"]]
                else:
                    hypotheses = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    n=n-existing_completions,
                    stop=stop,
                    max_tokens=max_tokens,
                    logprobs=1
                    )
                    result = [ (h["text"],
                                sum(h["logprobs"]["token_logprobs"]),
                                len(h["logprobs"]["token_logprobs"]))
                            for h in hypotheses["choices"]]
                break
            except (openai.error.RateLimitError, openai.error.Timeout):
                print(f"Exceeded rate limit, blocking {backoff}s", openai.api_key)
                time.sleep(backoff)
                backoff*=2
            except FunctionTimedOut:
                print(f"Timed out, blocking {backoff}s", openai.api_key)
                time.sleep(backoff)
                backoff*=2
            except (openai.error.APIError,openai.error.APIConnectionError,openai.error.ServiceUnavailableError):
                print("openai.error.APIError, blocking 10s")
                time.sleep(10)
            except openai.error.InvalidRequestError as e:
                print("openai.error.InvalidRequestError", e, engine)
                time.sleep(10)

        print("OpenAI responded!")
    
    print("Extending DB")

    completions_database.extend(prompt, engine, result, temperature, max_tokens, stop)

    # Now we fetch `n` completions from the database, and return them.
    return_value = completions_database.lookup(prompt, engine, temperature, max_tokens, stop, n=n)
    assert len(return_value) == n

    print("finished DB extension")

    return return_value

def batch_program_translations(prompts, stop="\n", max_tokens=128):
    global keys

    engine="code-davinci-002"
    temperature = 0

    # We first check the database to see if we have already computed any of the completions.
    
    existing_completions = [completions_database.n_entries(prompt, engine, temperature, max_tokens, stop)
                            for prompt in prompts]
    missing_prompts = [prompt for prompt, existing_completion in zip(prompts, existing_completions) if existing_completion < 1]

    if len(missing_prompts) > 20: # this is what OpenAI caps us at
        batch_program_translations(missing_prompts[:20], stop=stop, max_tokens=max_tokens)
        # now they won't be missing on the next time we call this function
        return batch_program_translations(prompts, stop=stop, max_tokens=max_tokens)

    if len(missing_prompts) > 0:
        print(f"extending cached completions: {len(missing_prompts)} new prompts")
        
        backoff=1
        print("Querying OpenAI...")
        
        while True:
            
            try:
                openai.api_key=keys[0]
                keys = keys[1:] + [keys[0]]
                
                hypotheses = openai.Completion.create(
                    engine=engine,
                    prompt=missing_prompts,
                    temperature=temperature,
                    n=1,
                    stop=stop,
                    max_tokens=max_tokens,
                    logprobs=1
                )
                result = [ (h["text"],
                            sum(h["logprobs"]["token_logprobs"]),
                            len(h["logprobs"]["token_logprobs"]))
                        for h in hypotheses["choices"]]
                break
            except (openai.error.RateLimitError, openai.error.Timeout):
                print(f"Exceeded rate limit, blocking {backoff}s", openai.api_key)
                time.sleep(backoff)
                backoff*=2
            except FunctionTimedOut:
                print(f"Timed out, blocking {backoff}s", openai.api_key)
                time.sleep(backoff)
                backoff*=2
            except (openai.error.APIError,openai.error.APIConnectionError,openai.error.ServiceUnavailableError):
                print("openai.error.APIError, blocking 10s")
                time.sleep(10)

        print("OpenAI responded!")
        print("Extending DB")

        for prompt, r in zip(missing_prompts, result):
            completions_database.extend(prompt, engine, [r], temperature, max_tokens, stop)

    # Now we fetch `n=1` completions from the database, and return them.
    return_value = [completions_database.lookup(prompt, engine, temperature, max_tokens, stop, n=1)[0]
                    for prompt in prompts]

    return return_value


def propose_code_hypotheses(examples, n, temperature=1):
    # https://dspace.mit.edu/bitstream/handle/1721.1/16714/42471842-MIT.pdf?sequence=2&isAllowed=y
    return sample_completions(
        engine="code-davinci-002",
        prompt=f"""# Python 3
# Here are a few example number concepts:
# -- The number is even
# -- The number is between 30 and 45
# -- The number is a power of 3
# -- The number is less than 10
# 
# Here are some random examples of numbers belonging to a different number concept:
# {', '.join(map(str,examples))}
# Write a python function that returns true if `num` belongs to this number concept.
def check_if_in_concept(num):
   return""",
        temperature=1,
        n=n,
        stop="\n",
        max_tokens=128
    )

def propose_nl_hypotheses(examples, n, temperature=1, engine="code-davinci-002"):
    # https://dspace.mit.edu/bitstream/handle/1721.1/16714/42471842-MIT.pdf?sequence=2&isAllowed=y
    if examples:
        return sample_completions(
            engine=engine,
            prompt=f"""# Python 3
# Here are a few example number concepts:
# -- The number is even
# -- The number is between 30 and 45
# -- The number is a power of 3
# -- The number is less than 10
# 
# Here are some random examples of numbers belonging to a different number concept:
# {', '.join(map(str,examples))}
# The above are examples of the following number concept:
# -- The number is """,
            temperature=1,
            n=n,
            stop="\n",
            max_tokens=128
        )
    else:
        return sample_completions(
            engine=engine,
            prompt=f"""# Python 3
# Here are a few example number concepts:
# -- The number is even
# -- The number is between 30 and 45
# -- The number is a power of 3
# -- The number is less than 10
# -- The number is """,
            temperature=1,
            n=n,
            stop="\n",
            max_tokens=128
        )

def shape_examples_prompt(positives, negatives=None, prompt_index=0):
    """returns a nice table showing some examples, useful for building prompts"""
    
    if negatives is None:
        examples = positives
        positives =  [ features for ex in examples for flag, features in ex if flag ]
        negatives =  [ features for ex in examples for flag, features in ex if not flag ]

        # reverse both of them so that the most recent example is first
        positives = positives[::-1]
        negatives = negatives[::-1]

        # remove duplicates
        positives = [*dict.fromkeys(positives)]
        negatives = [*dict.fromkeys(negatives)]

    domain = {"Size": [1, 2, 3], 
                "Shape": ["triangle", "rectangle", "circle"] if prompt_index%2 == 0 else ["circle", "triangle", "rectangle"], 
                "Color": ["yellow", "green", "blue"] if prompt_index%2 == 0 else ["blue", "yellow", "green"] }
    prompts = []
    for ordering in itertools.permutations(domain.keys()):
        examples = [ f"|{ordering[0]}|{ordering[1]}|{ordering[2]}|In concept?|", "---------" ]
        for feature_value_1 in domain[ordering[0]]:
            for feature_value_2 in domain[ordering[1]]:
                for feature_value_3 in domain[ordering[2]]:
                    features = (feature_value_1, feature_value_2, feature_value_3)
                    shape = features[ordering.index("Shape")]
                    color = features[ordering.index("Color")]
                    size = features[ordering.index("Size")]
                    key = (shape, color, size)
                    if (shape, color, size) in positives: judgment="yes"
                    elif (shape, color, size) in negatives: judgment="no"
                    else: judgment="maybe"
                    ssize = ["small", "medium", "large"][size-1]
                    if judgment != "maybe":
                        features = [ ["small", "medium", "large"][f-1] if isinstance(f, int) else f for f in features ]
                        examples.append(f"|{'|'.join(features)}|{judgment}|")
        examples = "\n".join(examples)
        prompts.append(examples)
    return prompts

_nl_set_hypothesis_cache = {}
def propose_nl_set_hypothesis(examples, n, prompt_index=0, temperature=1):
    """prompt_index, even/odd: toggles ordering of shape/color/size in the prompt
    prompt_index, 0-1:  give examples using the actual features
    prompt_index, 2-3:  give examples using the silly features (for example purple)
    prompt_index, 4-5: ask it to enumerate possible rules at zero temperature"""
    cache_key = (tuple(e for es in examples for e in es ), n)
    if cache_key in _nl_set_hypothesis_cache:
        return list(_nl_set_hypothesis_cache[cache_key])
    
    positives =  [ features for ex in examples for flag, features in ex if flag ]
    negatives =  [ features for ex in examples for flag, features in ex if not flag ]

    # reverse both of them so that the most recent example is first
    positives = positives[::-1]
    negatives = negatives[::-1]

    # remove duplicates
    positives = [*dict.fromkeys(positives)]
    negatives = [*dict.fromkeys(negatives)]

    def verbalize_features(f):
        shape, color, size = f
        size = ["small", "medium", "large"][size-1]
        return f"{size} {color} {shape}"

    if len(examples) == 0:
        prompt = """Here are some example concepts defined by a logical rule:

Rule: color is purple.
Rule: shape is not a hexagon.
Rule: color is purple and size is small.
Rule: size is tiny or shape is square.
Rule: size is not enormous.
Rule: color is red.

Please propose a some new concepts, defined by a logical rule. These new concepts can only refer to the following features:
- shape: triangle, rectangle, circle
- color: green, blue, yellow
- size: small, medium, large

Please make your rules short and simple, and please write your response on a single line that begins with the text "Rule: ". Provide 100 possible rules."""
        completions = sample_completions(prompt=prompt,
                                n=1,
                                engine="gpt-4", temperature=0, max_tokens=2048, stop="")[0]
        
        completions = [(" ".join(c.split(" ")[1:]), 0, 0) for c in completions[0].split("\n") if c.endswith(".") ][:n]
    else:
        if prompt_index in [2,3]:
            exemplar_rules = """Rule: color is purple.
Rule: shape is not a hexagon.
Rule: color is purple and size is small.
Rule: size is tiny or shape is square.
Rule: size is not enormous.
Rule: color is red."""
        else:
            exemplar_rules = """Rule: a triangle.
Rule: a green rectangle.
Rule: big or a rectangle (unless that rectangle is blue).
Rule: not both big and green.
Rule: either big or green, but not both.
Rule: either a rectangle or not yellow.
Rule: a circle."""

        completions = []
        for examples in shape_examples_prompt(positives, negatives, prompt_index):
            if prompt_index in [0,1,2,3]:
                prompt = f"""Here are some example concepts defined by a logical rule:

{exemplar_rules}

Now please produce a logical rule for a new concept. Your rule should be consistent with the following examples. It must be true of all the positive examples, and not true of all the negative examples. The examples are organized into a table with one column for each feature (size, color, shape):

{examples}

Please produce a simple rule that is consistent with the above table. Make your rule as SHORT, SIMPLE, and GENERAL as possible. Do NOT make it more complicated than it has to be, or refer to features that you absolutely do not have to refer to. Begin by writing "Rule: " and then the rule, followed by a period."""
           
                completions.extend(sample_completions(prompt=prompt,
                                    n=int(n/6 + 0.5), # 6 permutations of the features
                                    engine="gpt-4", #, #"code-davinci-002",#"gpt-3-turbo",
                                    temperature=temperature*0.99, # I don't know why I did this but I'm afraid change it
                                    stop="\n",
                                    max_tokens=100))
            elif prompt_index in [4,5]:
                prompt = f"""Here are some example concepts defined by a logical rule:

{exemplar_rules}

Now please produce possible logical rules for a new concept. Your rule should be consistent with the following examples. It must be true of all the positive examples, and not true of all the negative examples. The examples are organized into a table with one column for each feature (size, color, shape):

{examples}

Please produce {n+1} simple rules that are consistent with the above table. Each rule should be consistent with the examples, and should be as SHORT, SIMPLE, and GENERAL as possible. Do NOT make it more complicated than it has to be, or refer to features that you absolutely do not have to refer to. Begin each rule by writing "Rule: " and then the rule, followed by a period.
Number your rules starting at 1. For example, the first line that you produce should read "1. Rule: " and then the rule, followed by a period."""
                new_completions = sample_completions(prompt=prompt,
                                                n=1, 
                                                engine="gpt-4", #"gpt-3-turbo",
                                                temperature=0,
                                                stop=f"{n+1}.", max_tokens=1000)[0]
                new_completions = [ " ".join(proposal.split(" ")[1:]) for proposal in new_completions[0].split("\n") if len(proposal)>0 ]
                # print(examples)
                # print("\n".join(new_completions))
                completions.extend([ (c, 0, 0) for c in new_completions ])
            

    _nl_set_hypothesis_cache[cache_key] = completions
    return list(completions)

def higher_order_examples_text(examples):
    def verbalize_features(f):
        shape, color, size = f
        size = ["small", "medium", "large"][size-1]
        return f"({size} {color} {shape})"

    text_examples = []
    for i, ex in enumerate(examples):
        if i == 0:
            text_examples.append("    An Example of Concept #4:")
        else:
            text_examples.append("    Another Example of Concept #4:")
        positives =  [ verbalize_features(features) for flag, features in ex if flag ]
        negatives =  [ verbalize_features(features) for flag, features in ex if not flag ]
        if positives:
            text_examples.append(f"        POSITIVES: {', '.join(positives)}")
        else:
            text_examples.append(f"        POSITIVES: none")
        if negatives:
            text_examples.append(f"        NEGATIVES: {', '.join(negatives)}")
        else:
            text_examples.append(f"        NEGATIVES: none")
    examples = "\n".join(text_examples)
    return examples

def propose_nl_higher_order_set_hypothesis(examples, n, prompt_index=0, temperature=1):

    if len(examples) == 0:
        prompt = """Here are some example concepts defined by a logical rule:
1. Something is positive if it is the biggest yellow object in the example
2. Something is positive if there is another object with the same color in the example
3. Something is positive if it is the same color as the smallest triangle in the example

Please propose a new concept, defined by a logical rule. These new concepts can only refer to the following features:
- shape: triangle, rectangle, circle
- color: green, blue, yellow
- size: small, medium, large
Your rule can make comparisons between objects in the example, for instance, comparing sizes, colors, and shapes.

Please make your rule SHORT and SIMPLE. Start the rule with the text "Something is positive if".
Please provide a possible rule now."""
        completions = sample_completions(prompt=prompt,
                                n=n,
                                        engine="gpt-4", #"gpt-3.5-turbo", #"gpt-4", # "gpt-4", #"text-davinci-003", #"gpt-3.5-turbo", #"code-davinci-002",#"gpt-3-turbo",        
                        temperature=1,        
                                        stop="\n",
            max_tokens=32
        )
        
        return completions

    examples = higher_order_examples_text(examples)

    if prompt_index == 0:
        generation_instructions = "Infer ten different possible rules, and make those ten rules as simple and general as you can. Your simple general rules might talk about shapes, colors, and sizes, and might make comparisons between these features within a single example, but it doesn't have to. Remember that a rule should say when something is positive, and should mention the other objects in the example, and should be consisting with what you see below."
        generation_instructions2 = "Now make a numbered list of 10 possible rules for Concept #4. Start by writing \"1. Something is positive if\". End each line with a period."
        stop = ""
        max_tokens = 512
        assert temperature == 1
    if prompt_index == 1:
        generation_instructions = "Infer the rule for Concept #4, and make it as simple and general as you can. Your simple general rule might talk about shapes, colors, and sizes, and might make comparisons between these features within a single example, but it doesn't have to. Remember that the rule should say when something is positive, and should mention the other objects in the example, and should be consisting with what you see below."
        generation_instructions2 = "Now write the rule for Concept #4, on a single line, starting with the text \"Rule for Concept #4: Something is positive if\"."
        stop = ("\n", ".")
        max_tokens = 64
    if prompt_index >= 2:
        assert False, "prompt_index must be 0 or 1"

    prompt = f"""Here three simple concepts, which specify when an object is 'positive' relative to an example collection of other objects. Before giving the rule for each concept, we give examples of collections of objects, and which objects in the collection are 'positive'.

Concept #1:
    An Example of Concept #1:
        POSITIVES: (big yellow rectangle)
        NEGATIVES: (big green circle), (medium yellow rectangle)
    Another Example of Concept #1:
        POSITIVES: (medium yellow rectangle)
        NEGATIVES: (big red circle), (small green circle)
Rule for Concept #1: Something is positive if it is the biggest yellow object in the example.


Concept #2:
    An Example of Concept #2:
        POSITIVES: (small yellow circle), (medium yellow rectangle)
        NEGATIVES: (big green circle), (big blue rectangle)
    Another Example of Concept #2:
        POSITIVES: (big blue circle), (medium blue rectangle)
        NEGATIVES: (small green circle), (medium yellow rectangle), 
Rule for Concept #2: Something is positive if there is another object with the same color in the example.

Concept #3:
    An Example of Concept #3:
        POSITIVES: (small yellow circle), (medium yellow rectangle)
        NEGATIVES: (big green circle), (big blue rectangle)
    Another Example of Concept #3:
        POSITIVES: (small blue circle), (small blue triangle), (medium blue rectangle)
        NEGATIVES: (medium green triangle), (big yellow rectangle)
    Another Example of Concept #3:
        POSITIVES: (big red rectangle), (medium red rectangle), (big red triangle)
        NEGATIVES: (medium green triangle), (big yellow rectangle) 
Rule for Concept #3: Something is positive if it is the same color as the smallest triangle in the example.

Now here are some examples of another concept called Concept #4, but this time we don't know the rule. {generation_instructions}

Concept #4:
    {examples}
Rule for Concept #4: Something is positive if...

{generation_instructions2}"""

    completions = sample_completions(prompt=prompt,
                              n=10,
                                     engine="gpt-4", #gpt-3.5-turbo", #"gpt-4", # "gpt-4", #"text-davinci-003", #"gpt-3.5-turbo", #"code-davinci-002",#"gpt-3-turbo",        
        temperature=temperature,        
                                     stop=stop, 
        max_tokens=max_tokens
    )

    def postprocess0(proposal):
        proposal = " ".join(proposal.split(" ")[1:]).strip()
        # if it ends with a period, remove it
        if proposal.endswith("."):
            proposal = proposal[:-1]
        return proposal
    def postprocess1(proposal):
        return proposal.replace("Rule for Concept #4: ", "")
    postprocess = [postprocess0, postprocess1][prompt_index]

    for proposals, _, _ in completions:
        lines = [ln for ln in proposals.split("\n") if len(ln.strip()) > 0]
        n_missing_period = sum( not line.strip().endswith(".") for line in lines )
        n_lines = len(lines)
        # if n_missing_period > 0 or n_lines != 10:
        #     print("# proposals:", n_lines, "# missing period:", n_missing_period)
        #     print(proposals)
    
    possible_rules = [ [postprocess(proposal) for proposal in proposals.split("\n") if len(proposal.strip()) > 0] 
                        for proposals, _, _ in completions]

    return_value = []
    while len(return_value) < n and any( len(rule_list) > 0 for rule_list in possible_rules ):
        # round robin through the possible rules
        first_rule_list = possible_rules[0]
        rest_rule_list = possible_rules[1:]

        if len(first_rule_list) > 0:
            return_value.append(first_rule_list[0])
            first_rule_list = first_rule_list[1:]

        possible_rules = rest_rule_list + [first_rule_list]

    if len(return_value) < n:
        print("WARNING: not enough rules generated. This is not impossible but it should be rare.")

    return_value = { (r, 0, 0) for r in return_value }
    
    return list(return_value)
