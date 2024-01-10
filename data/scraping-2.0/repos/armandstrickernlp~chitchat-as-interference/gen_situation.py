import json
import os
import random
import argparse
import pprint as pp
from tqdm import tqdm


import torch
from datasets import Dataset
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          pipeline)
from transformers.pipelines.pt_utils import KeyDataset




def make_pipe(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, cache_dir="./llama2_cache")
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                use_auth_token=True, 
                                                cache_dir="./llama2_cache",
                                                device_map="auto",
                                                )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                    max_new_tokens = 250,
                    do_sample=False,
                    num_return_sequences=1,
                    return_full_text=False,
                    temperature=0,
                    )
    return pipe

def reformat_dial_names(dial_nums):
    return [x.replace(".json", "") for x in dial_nums]


def get_prep_exchanges(prep, dial_nums):
    """ get exchanges from prepended data that correspond to dial_nums"""
    prep_exchanges = {}
    for idx, dial_num in enumerate(prep):
        if dial_num in dial_nums:
            example = []
            for idx, (turn, typ) in enumerate(zip(prep[dial_num]["turns"], prep[dial_num]["types"])):
                if typ == "prepended" :#or typ == "rewritten":
                    if idx % 2 == 0:
                        example.append("User: "+turn.strip())
                    else:
                        example.append("System: "+turn.strip())
            example = example[:-1] # leqve out lat system turn for summarizing
            prep_exchanges[dial_num] = example
    return prep_exchanges


def make_sumSit_prompt():
    # create our examples
    # numbers : SNG02298, MUL2085, MUL2057
    sum_examples = [
        {
            "exchange": 
            """
    'User: I was talking to my brother on the phone earlier.',
    'System: What did you talk about with him?',
    'User: We were discussing about his wedding plans and decided to meet up at the london liverpool \
    street train station today.',
    'System: Do you need help getting there?'
    """,
            
            "situation": 
            """
    I was talking to my brother on the phone earlier today. He's getting married ! \
    We discussed his wedding plans and decided to meet up at the London Liverpool Street train station today. <end_situation>"""
        }, 
        
        {
            "exchange": 
            """
    'User: Recently, I have been given a long break by my boss to go on a holiday.',
    "System: That's great news! There's a plethora of holiday destination out there, have you made up your mind?',
    'User: I heard that Bishops Stortford is an ideal place for one to relax and enjoy.'
    """,
            
            "situation": 
            """
    I was recently given a long break by my boss to go on holiday. I heard that Bishops Stortford is an ideal place \
    for one to relax and enjoy. <end_situation>"""
        },

        {
            "exchange": 
            """
    'User: School is ending and it will be my vacation very soon!',
    'System: That sounds great, do you have any plans this summer?',
    'User: My friends and I have decided to meet at stansted airport and travel around.',
    'System: That sounds great, do you need help getting around?'
    """,
                
            "situation": 
            """
    School is ending very soon and I'll be on vacation! My friends and I have decided to meet at Stansted \
    Airport and travel around.<end_situation>"""
        }
    ]

    # create an example template
    sum_example_template = """
    Exchange: {exchange}
    Summarized situation: {situation}
    """

    # create a prompt example from above template
    sum_prompt = PromptTemplate(
        input_variables=["exchange", "situation"],
        template=sum_example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    sum_prefix = """The following are excerpts from conversations with an AI system. \
    From the exchange, the user's utterances have been summarized to describe the user's situation. \
    Importantly, information given by the user has not been changed, only condensed into a single passage, recounting the situation."""

    # and the suffix our user input and output indicator
    sum_suffix = """
    Exchange: {exchange}
    Summarized situation: 
    """

    # now create the few-shot prompt template
    few_shot_sum = FewShotPromptTemplate(
        examples=sum_examples,
        example_prompt=sum_prompt,
        prefix=sum_prefix,
        suffix=sum_suffix,
        input_variables=["exchange"],
        example_separator="\n"
    )

    return few_shot_sum


def parse_output_sum(text_out):
    return text_out.split("<end_situation>")[0]

def prep_sum_data(split_prep, few_shot_sum=make_sumSit_prompt()):
    gen_situations = {}
    dial_nums = []
    llm_input = {"text": []}
    for idx, dial_num in tqdm(enumerate(split_prep)):
        exchange = split_prep[dial_num]
        exchange = str(exchange).replace('[', '').replace(']', '').replace('", ', '",\n').replace("', ", "',\n")
        gen_situations[dial_num] = {}
        gen_situations[dial_num]["exchange"] = exchange

        txt_input = few_shot_sum.format(exchange=exchange)
        dial_nums.append(dial_num)
        llm_input["text"].append(txt_input)

    dataset = Dataset.from_dict(llm_input)

    return gen_situations, dial_nums, dataset


def generate_situations(gen_situations, dial_nums, dataset, pipe, bs=32):
    print("Generating situations...")
    gen_sit_list = []
    for out in tqdm(pipe(KeyDataset(dataset, "text"), batch_size=bs)):
        for gen in out:
            generated_sit = parse_output_sum(gen["generated_text"])
            gen_sit_list.append(generated_sit.strip())

    for idx, gen in enumerate(gen_sit_list):
        dial_num = dial_nums[idx]
        gen_situations[dial_num]["gen_sit"] = gen

    return gen_situations

        
# GENERATE CHITCHAT
def extract_domain_related_turns(mwoz):
    """ for each dailogue, extract turns that are from the domain relevant to the extracted situation.
        Cut off when there is a change in domain by looking at change in acts"""
    possible_user_acts = {
        'attraction',
        'hospital',
        'hotel',
        'police',
        'restaurant',
        'taxi',
        'train'}

    compatible_turns = {} # these are turns that are from the domain relevant to the extracted situation

    for i, dial_num in enumerate(mwoz):
        # original_dialogues = []
        selected_turns = []
        original_domain = None
        
        # for turn in mwoz[dial_num]["log"]:
        #     original_dialogues.append(f"{turn['text']} \t {[act.split('-')[0].lower() for act in turn['dialog_act']]}")
        
        stop_idx = 0
        for idx, turn in enumerate(mwoz[dial_num]["log"]):
            
            if "SNG" in dial_num:
                if idx % 2 == 0:
                    selected_turns.append(f'User: {turn["text"]}')
                else:
                    selected_turns.append(f'System: {turn["text"]}')
            
            else:
                # detect domain change if dialogue has multiple domains
                acts = [act.split('-')[0].lower() for act in turn["dialog_act"]]
                
                if original_domain is None : # set original domain 
                    if len(acts) == 1 and acts[0] != "general":
                        original_domain = acts[0]
                    elif len(acts) > 1:
                        for act in acts:
                            if act != "general":
                                original_domain = act
                                break
                    if idx % 2 == 0:
                        selected_turns.append(f'User: {turn["text"]}')
                    else:
                        selected_turns.append(f'System: {turn["text"]}')
                
                else : 
                    acts = set(acts) 
                    # if more than 1 act type that doesnt involve booking or a general comment, then there's a new domain
                    if len(acts) > 1 and "booking" not in acts and "general" not in acts:
                        stop_idx = idx
                        break
                    # if the original domain isnt there and the act type is from the possible user acts, then there's a new domain
                    elif original_domain not in acts and acts.issubset(possible_user_acts):
                        stop_idx = idx
                        break
                    else:
                        if idx % 2 == 0:
                            selected_turns.append(f'User: {turn["text"]}')
                        else:
                            selected_turns.append(f'System: {turn["text"]}')
            
        if len(selected_turns) == 1: 
            # these are examples with potential inconcistencies/errors in the acts
            # add the response to have at least a 1 full exchange
            selected_turns.append(f'System: {mwoz[dial_num]["log"][stop_idx]["text"]}')
                
        # make sure to end on a system response
        
        if len(selected_turns) % 2 == 1:
            selected_turns = selected_turns[:-1]

        # add example
        compatible_turns[dial_num.replace(".json", "")] = selected_turns
    
    return compatible_turns


def make_chitchat_prompt():
    # create our examples
    # leverage the situation + task history. try to make the added chitchat sound as natural and connected to the dialogue as possible
    # in the few shot examples
    examples = [
        {
            "situation": 
            """
    I was talking to my brother on the phone earlier today. He's getting married ! \
    We discussed his wedding plans and decided to meet up at the London Liverpool Street train station today.
    """,

            "context":
            """
    'User: I would like for a taxi to take me to london liverpool street train station, arriving no later than 17:45 please',
    "System: I can book that for you, first I'll need to know where you'd like picked up at.",
    """,

            "next_turns":
            """
    'User: I would like to depart from London Kings Cross Train Station <chitchat>.',
    'System: <chitchat> A white Toyota is booked for you.',
    """,

            "aug_next_turns":
            """
    "User: I would like to depart from London Kings Cross Train Station. \
    <My brother is getting married! I was talking to him on the phone earlier. London Liverpool train station is where we'll be meeting up actually.>",
    "System: <I see! Congratulations to your brother !> A white Toyota is booked for you."<end_example>
    """
        }, 
        {
            "situation": 
            """
    I was recently given a long break by my boss to go on holiday. \
    I heard that Bishops Stortford is an ideal place for one to relax and enjoy...
    """,

            "context":
            """
    'User: I need to find a train to bishops stortford please.',
    'System: When will you be departing and where will you be departing from?',
    'User: I need to leave from Cambridge and arrive by 14:30 in Bishops Stortford.',
    'System: There 35 trains to choose from, do you have a preference on a date and departure time?',
    """,

            "next_turns":
            """
    "User: Yes, I'd like to leave Thursday, what are the departure times and travel times? <chitchat>',
    'System: <chitchat> The first leaves at 05:29 and the last pulls out at 13:29. How many tickets please?'
    """,

            "aug_next_turns":
            """
    "User: Yes, I'd like to leave Thursday, what are the departure times and travel times? \
    <I have to say I'm really looking forward to traveling. I'm going on a holiday break and I heard Bishops Stortford is \
    ideal to relax and enjoy...>",
    "System: <Taking a break is definitely necessary sometimes. I'm certain you will find Bishops Stortford very enjoyable indeed! \
    Also,> the first leaves at 05:29 and the last pulls out at 13:29. How many tickets please?"<end_example>
    """
        },
            {
            "situation": 
            """
    School is ending very soon and I'll be on vacation! \
    My friends and I have decided to meet at Stansted Airport and travel around.
    """,

            "context":
            """
    'User: i need a train from stansted airport to cambridge',
    'System: What day and approximately what time will you be traveling to Cambridge?',
    'User: I need to get there by 20:30 on Wednesday.',
    'System: You could grab the TR3828 leaving stansted airport it will get you there by 09:52.',
    'User: Can you get me there a little bit earlier?',
    'System: How about 05:24 that is the earliest, how many tickets please?',
    """,

            "next_turns":
            """
    'User: There will be 8 of us. <chitchat>',
    'System: <chitchat> Okay, I have you booked for train TR5910. The total fee is 80.8 GDP.
    """,

            "aug_next_turns":
            """
    "User: There will be 8 of us. <I'm meeting my friends at the airport. We're going to be traveling around after school ends.>",
    "System: <That sounds like a fun trip ! I hope you all enjoy your visit to Cambridge.> \
    I have you booked for train TR5910. The total fee is 80.8 GDP."<end_example>
    """
        },
                
    ]

    # create an example template
    example_template = """
    Situation: {situation}
    Conversational Context: {context}
    Original Next Turns: {next_turns}
    Next Turns With Chitchat: {aug_next_turns}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["situation", "context", "next_turns", "aug_next_turns"],
        template=example_template
    )


    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """In the following examples, you are presented with a user's situation that only the user knows about,\
    and a conversational context, which may be empty if it is the start of the conversation. \
    The goal is to augment the original next turns with chitchat. \
    The user's chitchat should be contextual to the situation and previous conversational context. \
    It should re-introduce the situation for the system, as the system has no knowledge of the situation. \
    The system's chitchat should naturally lead back to the orginal response and should be unique to \
    to the conversational context. Only use statements and avoid repeating expressions found in previous examples. \
    The system chitchat should only display general knowledge and admit uncertainty when asked questions \
    about specific entities that would require an internet search. \
    Responses should be coherent and fluent.
    """
    # watch for:
    # - dial that uses encyclopedic knowledge and would need fact checking : kung pao chicken
    # - response that asks questions in the chitchat 
    # - imitates pretty closely the prompt examples for the system responses, especially when sample=False, get more unique responses
    # - sometimes the situation is taken for granted and the chitchat starts off with I hope he will... who is he ?
    
    suffix = """
    Situation: 
    {situation}
    Conversational Context: 
    {context}
    Original Next Turns: 
    {next_turns}
    Next Turns With Chitchat: 
    """

    # now create the few-shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["situation", "context", "next_turns"],
        example_separator="\n---"
    )
    return few_shot_prompt_template


def parse_output_chitchat(text_out):
    return text_out.split("<end_example>")[0]

def prep_chitchat_data(compatible_turns, gen_situations, few_shot_prompt_template=make_chitchat_prompt()):
    gen_chitchat = {}
    dial_nums = []
    llm_input = {"text": []}
    
    for i, dial_num in enumerate(gen_situations): 
        dial_nums.append(dial_num)
        gen_chitchat[dial_num] = {}
        
        sit = gen_situations[dial_num]["gen_sit"]
        gen_chitchat[dial_num]["situation"] = sit

        history = compatible_turns[dial_num]
        rand_turn = random.randint(0, len(history)-1)
        if rand_turn % 2 == 1:
            rand_turn -= 1
        context, next_turns = history[:rand_turn], history[rand_turn:rand_turn+2]
        context = str(context).replace('[', '').replace(']', '').replace('", ', '",\n').replace("', ", "',\n") if context != [] else "None\n"
        gen_chitchat[dial_num]["context"] = context

        next_turns[0] += " <chitchat>"
        next_turns[1] = next_turns[1].replace("System: ", "System: <chitchat> ")
        next_turns = str(next_turns).replace('[', '').replace(']', '').replace('", ', '",\n').replace("', ", "',\n") + "\n"
        gen_chitchat[dial_num]["next_turns"] = next_turns
        
        txt_input = few_shot_prompt_template.format(situation=sit, context=context, next_turns= next_turns)
        llm_input["text"].append(txt_input)

    dataset = Dataset.from_dict(llm_input)    
    return gen_chitchat, dial_nums, dataset


def generate_chitchat(gen_chitchat, cc_dial_nums, cc_dataset, pipe, bs=16):
    print("Generating chitchat...")
    gen_cc_list = []
    for out in tqdm(pipe(KeyDataset(cc_dataset, "text"), batch_size=bs)):
        for gen in out:
            chitchat_gen = parse_output_chitchat(gen["generated_text"])
            gen_cc_list.append(chitchat_gen.strip())
            
    for idx, gen in enumerate(gen_cc_list):
        dial_num = cc_dial_nums[idx]
        gen_chitchat[dial_num]["next_turns_chitchat"] = gen
    
    return gen_chitchat



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--data_split", type=str, default="valid", help="one of train, valid, test")
    parser.add_argument("--training_batch_number", type=int, default=0, help='one of 0,1,2 for train split') # run each batch in parallel vs fulll training set in one job
    args = parser.parse_args()

    pipe = make_pipe(model_name=args.model_name)

    # load data
    with open("data/fusedchat_prepended.json") as f:
        prep = json.load(f)

    with open('data/MultiWOZ_2.2.json') as f:
        mwoz = json.load(f)
    
    out_path = f"outputs/{args.data_split.upper()}"
    os.makedirs(out_path, exist_ok=True)
    
    # select dial_nums to use
    with open('data/valListFile.txt') as f:
        val_nums = f.read().splitlines()
    with open('data/testListFile.txt') as f:
        test_nums = f.read().splitlines()
    
    train_nums = [num for num in list(mwoz.keys()) if num not in val_nums and num not in test_nums]
    
    if args.data_split == "train":
        dial_nums = train_nums
    
    elif args.data_split == "valid":
        dial_nums = val_nums

    elif args.data_split == "test":
        dial_nums = test_nums

    dial_nums = reformat_dial_names(dial_nums)


    # GENERATE SUMMARIZED SITUATIONS
    split_prep = get_prep_exchanges(prep, dial_nums)
    if args.data_split == "train": # generate via different jobs
        train_nums = list(split_prep.keys())
        train_nums.sort()
        batch_size = len(train_nums) // 3
        train_nums_list = [train_nums[i:i + batch_size] for i in range(0, len(train_nums), batch_size)]
        if len(train_nums_list) == 4:
            train_nums_list[2] = train_nums_list[2] + train_nums_list[3]
            train_nums_list.pop()
        train_nums = train_nums_list[args.training_batch_number]
        split_prep = {k: split_prep[k] for k in train_nums}
        out_path = f"outputs/{args.data_split.upper()}_batch{args.training_batch_number}"
        os.makedirs(out_path, exist_ok=True)

    # FOR TESTING###############################################
    # split_prep = {k: split_prep[k] for k in list(split_prep.keys())[:5]}
    ############################################################
    print(f"Number of examples: {len(split_prep)}")

    few_shot_sum = make_sumSit_prompt()
    initial_gen_situations, sum_dial_nums, sum_dataset = prep_sum_data(split_prep, few_shot_sum=few_shot_sum)
    gen_situations = generate_situations(initial_gen_situations, sum_dial_nums, sum_dataset, pipe)

    with open(os.path.join(out_path, 
                          "gen_situations.json"), "w") as f:
        json.dump(gen_situations, f, indent=2)
    
  


    








