# -*- coding: utf-8 -*-
import time
import json
import os
import random
import argparse
from tqdm import tqdm
from chatarena.agent import Player, Moderator
from chatarena.backends import OpenAIChat
from chatarena.environments.conversation import ModeratedConversation
from chatarena.arena import Arena
from data_utils import find_word_in_string
from instruction import create_instruct


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cached_seed_path", type=str, required=True, 
                        help="The cached seed dialog file.")
    parser.add_argument("--profile_path", type=str, default="seed_dataset/caches/db_slot/slot_profiles.json", 
                        help="The user profile slot-values file.")
    parser.add_argument("--output_dir", type=str, default="data/TopDial", 
                        help="The output directory to save the simulated dialog data.")
    parser.add_argument("--max_interaction_step", type=int,default=12, 
                        help="The max number of interaction steps, i.e., 2 * max rounds.")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", 
                        help="The chat model to use.")
    parser.add_argument("--temperature", type=float, default=0.75, 
                        help="The temperature to use in sampling.")
    parser.add_argument("--max_system_tokens", type=int, default=100, 
                        help="The max number of tokens to generate for the system.")
    parser.add_argument("--max_user_tokens", type=int, default=80,
                        help="The max number of tokens to generate for the user.")
    parser.add_argument("--max_moderator_tokens", type=int, default=10,
                        help="The max number of tokens to generate for the moderator.")
    parser.add_argument("--show_description", type=str2bool, default="true", 
                        help="Whether to show the role description.")
    parser.add_argument("--show_message", type=str2bool, default="true", 
                        help="Whether to show the conversation messages.")
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def clean_utterance(s):
    s = s.strip()
    for start_str in ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]']:
        if s.startswith(start_str):
            s = s[len(start_str):].strip()
    return s

def prompt_conversation(raw_goal, conversation):
    """Prompt the conversation context."""
    conversation_ctx = ""
    for idx, utt in enumerate(conversation):
        utt = clean_utterance(utt)
        if "User Initiative" in raw_goal:
            if idx % 2 == 0:
                conversation_ctx += f"[Role-U]: {utt}<EOS>\n\n"
            else:
                conversation_ctx += f"[Role-S]: {utt}<EOS>\n\n"
        else:
            if idx % 2 == 0:
                conversation_ctx += f"[Role-S]: {utt}<EOS>\n\n"
            else:
                conversation_ctx += f"[Role-U]: {utt}<EOS>\n\n"
    return conversation_ctx

def sample_seed_conversation(raw_goal, conversation):
    """Sample seed conversations (continue | end)."""
    conv_lens = len(conversation)
    continue_len = random.choice(range(1, int(conv_lens * 0.6)))
    conv_continue = prompt_conversation(raw_goal, conversation[:continue_len])
    conv_end = prompt_conversation(raw_goal, conversation)
    seed_conv = {
        "seed_continue": conv_continue,
        "seed_end": conv_end
    }
    return seed_conv

def sample_assistant_role(profile_slots, user_profile):
    """Sample an assistant role."""
    all_names = profile_slots["Name"]
    user_name = user_profile["Name"]
    sampled_name = random.choice(all_names)
    while find_word_in_string(sampled_name, user_name):
        sampled_name = random.choice(all_names)
    return sampled_name

def sample_personality():
    """Sample a personality based on Big Five personality traits."""
    personalities = {
        "agreeableness": ["trustworthy, straightforward, and generous", "unreliable, complicated, meager, and boastful"],
        "conscientiousness": ["efficient, organized, and careful", "inefficient, careless, and sloppy"],
        "extraversion": ["outgoing, energetic, and talkative", "shy, reserved, and quiet"],
        "neuroticism": ["sensitive, nervous, and insecure", "secure, confident, and calm"],
        "openness": ["intellectual, imaginative, and curious", "unimaginative, uncreative, and conventional"]
    }
    sampled_personality = {}
    for trait, values in personalities.items():
        sampled_personality[trait] = random.choice(values)
    return sampled_personality


def generate_dialog_data(
    profile_path,
    seed_path,
    output_dir,
    max_interaction_step=10,
    model_name="gpt-3.5-turbo",
    temperature=0.75,
    max_system_tokens=100,
    max_user_tokens=80,
    max_moderator_tokens=10,
    show_description=True,
    show_message=True,
):
    """Generate dialog data from a seed dialog file."""
    profile_slots = json.load(open(profile_path, "r", encoding='utf-8'))
    print(f"Loaded user profiles with {len(profile_slots)} slot keys.")

    seed_dialogs = []
    with open(seed_path, "r", encoding='utf-8') as f:
        for line in f:
            seed_dialogs.append(json.loads(line))
    print(f"Loaded {len(seed_dialogs)} cached dialogs.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if "test_seen" in seed_path:
        output_path = os.path.join(output_dir, "dialogue_test_seen.jsonl")
    elif "test_unseen" in seed_path:
        output_path = os.path.join(output_dir, "dialogue_test_unseen.jsonl")
    elif "dev" in seed_path:
        output_path = os.path.join(output_dir, "dialogue_dev.jsonl")
    else:
        output_path = os.path.join(output_dir, "dialogue_train.jsonl")
    
    with open(output_path, "w", encoding='utf-8') as fw:
        for seed_dialog in tqdm(seed_dialogs):
            simulated_profile = seed_dialog["user_profile"]
            sampled_knowledge = seed_dialog["knowledge"]
            target = seed_dialog["target"]

            conversation = seed_dialog["seed_conversation"]
            seed_conv = sample_seed_conversation(seed_dialog["original_goal"], conversation)
            
            # randomly sample a personality
            simulated_personality = sample_personality()
            assistant_name = sample_assistant_role(profile_slots, simulated_profile)
            
            env_desc, user_dict, assistant_dict, moderator_dict = create_instruct(
                target=target,
                simulated_profile=simulated_profile,
                simulated_personality=simulated_personality,
                assistant_name=assistant_name,
                domain_knowledge=sampled_knowledge,
                seed_conversation=seed_conv
            )
            assistant = Player(
                name=assistant_dict["name"], backend=OpenAIChat(model=model_name, temperature=temperature, max_tokens=max_system_tokens),
                role_desc=assistant_dict["role_desc"], global_prompt=env_desc
            )
            user = Player(
                name=user_dict["name"], backend=OpenAIChat(model=model_name, temperature=temperature, max_tokens=max_user_tokens),
                role_desc=user_dict["role_desc"], global_prompt=env_desc
            )
            moderator = Moderator(
                backend=OpenAIChat(model=model_name, temperature=temperature, max_tokens=max_moderator_tokens),
                role_desc=moderator_dict["role_desc"], terminal_condition=moderator_dict["terminal_condition"]
            )
            # let assistant start the conversation
            env = ModeratedConversation(player_names=[p.name for p in [assistant, user]], moderator=moderator, moderator_period="round")
            arena = Arena(players=[assistant, user], environment=env, global_prompt=env_desc)
            
            arena.launch_cli(max_steps=max_interaction_step, show_description=show_description, show_message=show_message, interactive=False)

            #print("Save? (y/n)")
            #if input() == "n":
            #    continue
            
            # save the simulated dialog to file
            messages = env.get_observation()
            simulated_convs = []
            for msg in messages:
                if msg.agent_name == assistant.name:
                    utt = {"system": msg.content}
                else:
                    utt = {"user": msg.content}
                simulated_convs.append(utt)
            
            write_line = {
                "id": "s_" + str(seed_dialog["id"]),
                "user_profile": simulated_profile,
                "user_personality": simulated_personality,
                "knowledge": sampled_knowledge,
                "target": target,
                "conversation": simulated_convs
            }
            fw.write(json.dumps(write_line, ensure_ascii=False) + "\n")
            fw.flush()

            print("Sleeping for 5 seconds...")
            time.sleep(5)

            #print("Continue? (y/n)")
            #if input() == "n":
            #    break


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.random_seed)

    generate_dialog_data(args.profile_path, args.cached_seed_path, args.output_dir, 
                        max_interaction_step=args.max_interaction_step,
                        model_name=args.model_name,
                        temperature=args.temperature,
                        max_system_tokens=args.max_system_tokens,
                        max_user_tokens=args.max_user_tokens,
                        max_moderator_tokens=args.max_moderator_tokens,
                        show_description=args.show_description,
                        show_message=args.show_message)
