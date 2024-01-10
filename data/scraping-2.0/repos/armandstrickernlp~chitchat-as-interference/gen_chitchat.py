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


def make_userBackstory_prompt():
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

            "user_utt":
            """
    'User: **I would like to depart from London Kings Cross Train Station.**'
    """,

            "user_utt_with_backstory":
            """
    "User: **I would like to depart from London Kings Cross Train Station.** +  \
    <Backstory: My brother is getting married! I was talking to him on the phone earlier and we decided to meet at the London Liverpool train station.>" [END]
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

            "user_utt":
            """
    "User: **Yes, I'd like to leave Thursday, what are the departure times and travel times?**'
    """,

            "user_utt_with_backstory":
            """
    "User: **Yes, I'd like to leave Thursday, what are the departure times and travel times?** + \
    <Backstory: I have to say I'm really looking forward to going on holiday. I'm taking a break and I heard Bishops Stortford is \
    ideal to relax and enjoy...>" [END]
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
            "user_utt":
            """
    'User: **There will be 8 of us.**'""",

            "user_utt_with_backstory":
            """
    "User: **There will be 8 of us.** + <Backstory: I'm actually meeting my friends at the airport. We're going to be traveling around after school ends.>" [END]
    """,

        },
                
    ]

    # create an example template
    example_template = """
    User Situation: {situation}
    Conversational Context: {context}
    Original User Utterance: {user_utt}
    User Utterance With Backstory: {user_utt_with_backstory}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["situation", "context", "user_utt", "user_utt_with_backstory"],
        template=example_template
    )

    prefix = """In the following examples, you are presented with a user's situation and a conversational context, \
    which may be None if it is the start of the conversation. \
    The user shares their backstory by adding it to their original utterance. Their backstory is based on the user's situation and \
    should naturally follow the original utterance. It should be very fluent and coherent with the conversational context."""

 
    suffix = """
    Situation: 
    {situation}
    Conversational Context: 
    {context}
    Original User Utterance: 
    {user_utt}
    User Utterance With Backstory:
    """

    # now create the few-shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["situation", "context", "user_utt"],
        example_separator="\n"
    )
    return few_shot_prompt_template




def parse_output(text_out):
    return text_out.split(" [END]")[0]

def prep_usrBack_data(compatible_turns, gen_situations, few_shot_prompt_template=make_userBackstory_prompt()):
    gen_usr_back = {}
    dial_nums = []
    llm_input = {"text": []}
    
    for i, dial_num in enumerate(gen_situations): 
        dial_nums.append(dial_num)
        gen_usr_back[dial_num] = {}
        
        sit = gen_situations[dial_num]["gen_sit"]
        gen_usr_back[dial_num]["situation"] = sit

        history = compatible_turns[dial_num]
        rand_turn = random.randint(0, len(history)-1)
        if rand_turn % 2 == 1:
            rand_turn -= 1
        gen_usr_back[dial_num]["rand_idx"] = rand_turn

        context, usr_utt, sys_resp = history[:rand_turn], history[rand_turn:rand_turn+1], history[rand_turn+1:rand_turn+2]
        
        context = str(context).replace('[', '').replace(']', '').replace('", ', '",\n').replace("', ", "',\n") if context != [] else "None\n"
        gen_usr_back[dial_num]["context"] = context

        usr_utt = str(usr_utt).replace('[', '').replace(']', '')
        usr_utt = usr_utt.replace('User: ', 'User: **')
        usr_utt = usr_utt[:-1] + '**' + usr_utt[-1]
        gen_usr_back[dial_num]["usr_utt"] = usr_utt
        
        sys_resp = str(sys_resp).replace('[', '').replace(']', '')
        sys_resp = sys_resp.replace('System: ', 'System: **')
        sys_resp = sys_resp[:-1] + '**' + sys_resp[-1]
        gen_usr_back[dial_num]["sys_resp"] = sys_resp
        
    
        txt_input = few_shot_prompt_template.format(situation=sit, context=context, user_utt=usr_utt)
        llm_input["text"].append(txt_input)
       
    dataset = Dataset.from_dict(llm_input)    
    return gen_usr_back, dial_nums, dataset

def generate_usr_back(gen_usr_back, back_dial_nums, back_dataset, pipe, bs=16):
    print("Generating backstories...")
    gen_back_list = []
    for out in tqdm(pipe(KeyDataset(back_dataset, "text"), batch_size=bs)):
        for gen in out:
            backstory_gen = parse_output(gen["generated_text"])
            gen_back_list.append(backstory_gen.strip())
            
    for idx, gen in enumerate(gen_back_list):
        dial_num = back_dial_nums[idx]
        gen_usr_back[dial_num]["utt_with_backstory"] = gen
        
    return gen_usr_back


def make_RespReaction_prompt():
    examples = [
        {
            "context":
            """
    'User: I would like for a taxi to take me to london liverpool street train station, arriving no later than 17:45 please',
    "System: I can book that for you, first I'll need to know where you'd like picked up at.",
    "User: I would like to depart from London Kings Cross Train Station. <Backstory: My brother is getting married! \
    I was talking to him on the phone earlier and we decided to meet at the London Liverpool train station.>"
    """,

            "response":
            """
    'System: **A white Toyota is booked for you.**'
    """,

            "response_with_reaction":
            """
    "System: <Reaction: I see! Congratulations to your brother!> + **A white Toyota is booked for you.**" [END]
    """
        }, 
        {

            "context":
            """
    'User: I need to find a train to bishops stortford please.',
    'System: When will you be departing and where will you be departing from?',
    'User: I need to leave from Cambridge and arrive by 14:30 in Bishops Stortford.',
    "User: Yes, I'd like to leave Thursday, what are the departure times and travel times? \
    <Backstory: I have to say I'm really looking forward to going on holiday. I'm taking a break and I heard Bishops Stortford is \
    ideal to relax and enjoy...>',
    """,

            "response":
            """
    "System: **The first leaves at 05:29 and the last pulls out at 13:29. How many tickets please?**'
    """,

            "response_with_reaction":
            """
    "System: <Reaction: Taking a break is definitely necessary sometimes... I'm certain you will find Bishops Stortford very enjoyable indeed!> \
    + **The first leaves at 05:29 and the last pulls out at 13:29. How many tickets please?**" [END]
    """

        },
            {

            "context":
            """
    'User: i need a train from stansted airport to cambridge',
    'System: What day and approximately what time will you be traveling to Cambridge?',
    'User: I need to get there by 20:30 on Wednesday.',
    'System: You could grab the TR3828 leaving stansted airport it will get you there by 09:52.',
    'User: Can you get me there a little bit earlier?',
    'System: How about 05:24 that is the earliest, how many tickets please?',
    "User: There will be 8 of us. <Backstory: I'm actually meeting my friends at the airport. We're going to be traveling around after school ends.>"
    """,

            "response":
            """
    'System: **Okay, I have you booked for train TR5910. The total fee is 80.8 GDP.**'
    """,


            "response_with_reaction":
            """
    "System: <Reaction: That sounds like a fun trip ! I hope you all enjoy your visit to Cambridge.> \
    + **I have you booked for train TR5910. The total fee is 80.8 GDP.**" [END]
    """
        },

    ]

    # create an example template
    example_template = """
    Conversational Context: {context}
    Original System Response: {response}
    Response With Reaction: {response_with_reaction}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["context", "response", "response_with_reaction"],
        template=example_template
    )


    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions

    prefix = """In the following examples, you are presented with a conversational context. In the last turn, the user \
shares their backstory. The original system response should be improved to include a reaction to the user's backstory \
at the beginning of the response. This reaction should be supportive and display an understanding of the user's situation. \
It should be unique to the backstory and contextual to the conversational context. Avoid repeating expressions found in previous examples."""

    suffix = """
    Conversational Context: 
    {context}
    Original System Response: 
    {response}
    Response With Reaction: 
    """

    # now create the few-shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["context", "response"],
        example_separator="\n"
    )
    
    return few_shot_prompt_template



def prep_sysReac_data(gen_usr_back, few_shot_prompt_template=make_RespReaction_prompt()):
    gen_sys_react = {}
    dial_nums = []
    llm_input = {"text": []}
    
    for i, dial_num in enumerate(gen_usr_back): 
        dial_nums.append(dial_num)
        gen_sys_react[dial_num] = {}
        
        # copy over most fields
        gen_sys_react[dial_num]["situation"] = gen_usr_back[dial_num]["situation"]
        gen_sys_react[dial_num]["rand_idx"] = gen_usr_back[dial_num]["rand_idx"]
        gen_sys_react[dial_num]["context"] = gen_usr_back[dial_num]["context"]
        gen_sys_react[dial_num]["usr_utt"] = gen_usr_back[dial_num]["usr_utt"]
        gen_sys_react[dial_num]["sys_resp"] = gen_usr_back[dial_num]["sys_resp"]
        gen_sys_react[dial_num]["utt_with_backstory"] = gen_usr_back[dial_num]["utt_with_backstory"]
        
        
        # modify context to add generated user utterance
        context = gen_usr_back[dial_num]["context"]
        aug_usr_utt = gen_usr_back[dial_num]["utt_with_backstory"].replace("**", "").replace("+", "")
        
        if context == 'None\n':
            context = aug_usr_utt
        else:
            context = context + '\n' + aug_usr_utt
        
        sys_resp = gen_usr_back[dial_num]["sys_resp"]
    
        txt_input = few_shot_prompt_template.format(context=context, response=sys_resp)
        llm_input["text"].append(txt_input)
       
    dataset = Dataset.from_dict(llm_input)    
    return gen_sys_react, dial_nums, dataset

def generate_sys_resp(gen_sys_react, dial_nums, dataset, pipe, bs=10):
    print("Generating Reactions...")
    gen_reacts = []
    for out in tqdm(pipe(KeyDataset(dataset, "text"), batch_size=bs)):
        for gen in out:
            react_gen = parse_output(gen["generated_text"])
            gen_reacts.append(react_gen.strip())
            
    for idx, gen in enumerate(gen_reacts):
        dial_num = dial_nums[idx]
        gen_sys_react[dial_num]["resp_with_reaction"] = gen
        
    return gen_sys_react


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--gen_sit_path", type=str, default="outputs/VALID/gen_situations.json", help="path to generated situations")
    parser.add_argument("--gen_back_path", type=str, default="outputs/VALID/gen_backstories.json", help='path to generated backstories if they already exist')
    args = parser.parse_args()

    pipe = make_pipe(model_name=args.model_name)

    # load data
 
    with open('data/MultiWOZ_2.2.json') as f:
        mwoz = json.load(f)
    
    out_path = f"outputs/{args.gen_sit_path.split('/')[1]}"
    print(f"Saving backstories to {out_path}")
    os.makedirs(out_path, exist_ok=True)
    
    print(f"Loading generated situations from {args.gen_sit_path}")
    with open(args.gen_sit_path) as f:
        gen_situations = json.load(f)

    # GENERATE BACKSTORIES
    compatible_turns = extract_domain_related_turns(mwoz)
    random.seed(42) # turns to augment are selected randomly within the compatible turns
    gen_usr_back, back_dial_nums, back_dataset = prep_usrBack_data(compatible_turns, gen_situations)
    final_gen_usr_back = generate_usr_back(gen_usr_back, back_dial_nums, back_dataset, pipe)
    with open(os.path.join(out_path,
                           "gen_backstories.json"), "w") as f:
        json.dump(final_gen_usr_back, f, indent=2)

    out_path = f"outputs/{args.gen_back_path.split('/')[1]}"
    print(f"Saving reactions to {out_path}")
    os.makedirs(out_path, exist_ok=True)

    # print(f"Loading generated backstories from {args.gen_back_path}")
    # with open(args.gen_back_path) as f:
    #     final_gen_usr_back = json.load(f)

    #GENERATE REACTIONS
    gen_sys_react, dial_nums, dataset = prep_sysReac_data(final_gen_usr_back)
    final_gen_sys_react = generate_sys_resp(gen_sys_react, dial_nums, dataset, pipe)
    with open(os.path.join(out_path,
                           "gen_reactions.json"), "w") as f:
        json.dump(final_gen_sys_react, f, indent=2)







    








