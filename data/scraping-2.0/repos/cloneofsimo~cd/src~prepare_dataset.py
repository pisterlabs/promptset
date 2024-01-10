from typing import List, Type
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM


# you can prepare CSV dataset from sample_from_model
# https://www.reddit.com/r/LocalLLaMA/comments/155po2p/get_llama_2_prompt_format_right/
# hmm https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5

HOMAR_CONTEXT_EXAMPLE = """<s>[INST] <<SYS>>
You are a helpful agent that talks like Homar Simpson. Answer to users query like you are Homar Simpson. For example, here is the way he might talk:

'I've learned that life is one crushing defeat after another until you just wish Flanders was dead.'

'A gun is not a weapon, it's a tool, like a hammer or a screwdriver or an alligator.'

'Weaseling out of things is important to learn; it's what separates us from the animals… except the weasel.'

'Operator! Give me the number for 911!'

'If he's so smart, how come he's dead?'

'Marge, you know it's rude to talk when my mouth is full.'

'My beer! You never had a chance to become my urine!'

'Stupidity got us into this mess, and stupidity will get us out.'

'Trying is the first step towards failure.'

'Oh yeah, what are you gonna do? Release the dogs? Or the bees? Or the dogs with bees in their mouths and when they bark, they shoot bees at you?'


'Kids, just because I don't care doesn't mean I'm not listening.'

'I wish God were alive to see this.'

'Roads are just a suggestion Marge, just like pants.'

'We can outsmart those dolphins. Don't forget - we invented computers, leg warmers, bendy straws, peel-and-eat shrimp, the glory hole, and the pudding cup.'

'If it doesn't have Siamese twins in a jar, it is not a fair.'

'I'm like that guy who single-handedly built the rocket & flew to the moon. What was his name? Apollo Creed?'

'If God didn't want me to eat chicken in church, then he would have made gluttony a sin.'

'Volunteering is for suckers. Did you know that volunteers don't even get paid for the stuff they do?'

'Just sit through this NRA meeting Marge, and if you still don't think guns are great then we'll argue some more.'

'When will I learn? The answer to life's problems aren't at the bottom of a bottle, they're on TV!'

'Kids are great. You can teach them to hate what you hate and, with the Internet and all, they practically raise themselves.'

'Oh, I have three kids and no money. Why can't I have no kids and three money?'

'I want to share something with you: The three little sentences that will get you through life. Number 1: Cover for me. Number 2: Oh, good idea, Boss! Number 3: It was like that when I got here.'

'I'm normally not a praying man, but if you're up there, please save me Superman.'

Based on these example quotes from Homar Simpson, talk back to user like you are Homar Simpson.
<</SYS>>
{} [/INST] {}
"""


def sample_from_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_samples: int = 100,
    input_prompt_template: str = HOMAR_CONTEXT_EXAMPLE,
    input_queries: List[str] = None,
    **kwargs,
):
    outputs = []
    for queries in input_queries:
        input_prompt = input_prompt_template.format(queries, "")
        input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(model.device)
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=2048,
            temperature=0.7,
            top_k=0,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=n_samples,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        for i, output_sequence in enumerate(output_sequences):
            # print(f"{i}: {tokenizer.decode(output_sequence)}")
            # print just the text without the input prompt
            decoded = tokenizer.decode(
                output_sequence[
                    len(tokenizer.encode(input_prompt, return_tensors="pt")[0]) :
                ],
                skip_special_tokens=True,
            )
            print(f"Output for {i}: ", decoded)
            print(len(decoded))
            outputs.append(decoded)

    return outputs


# Example of using OpenAI API to generate dataset

def sample_from_openai_api(
    input_prompt_template: str = HOMAR_CONTEXT_EXAMPLE,
    input_queries: List[str] = None,
):
    try:
        import openai
        
    except ImportError:
        print("Please install openai library")
        return
    
    outputs = []
    
    for queries in input_queries:
        input_prompt = input_prompt_template.format(queries, "")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "text": input_prompt_template.format(queries, ""),
                },
            ],
        )
        
        print(response.choices[0]['message']['content'])
        print(len(response.choices[0]['message']['content']))

        outputs.append(response.choices[0]['message']['content'])
        
    return outputs
        

if __name__ == "__main__":
    example_queries = [
        "What is the meaning of life?",
        "What is the best way to make money?",
        "How do I get a girlfriend?",
        "What do you think about the meaning of life?",
        "What should I eat for dinner?",
        "Think of a question you want to ask Homar Simpson and ask it here.",
    ]

    model_name = "meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    N = 3
    outputs = sample_from_model(
        model=model,
        tokenizer=tokenizer,
        input_queries=example_queries,
        n_samples=N,
    )

    # make csv with pandas

    import pandas as pd

    df = pd.DataFrame(columns=["query", "response"])

    # time n
    queries = []
    for j in range(N):
        for i in range(len(example_queries)):
            queries.append(example_queries[i])

    df["query"] = queries
    df["response"] = outputs

    df.to_csv("homar_simpson.csv", index=False)
