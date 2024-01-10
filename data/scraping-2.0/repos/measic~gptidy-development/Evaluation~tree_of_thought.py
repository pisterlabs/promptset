# Adapted from research paper on Tree of Thought
import re
import openai

# set key
openai.api_key = "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"

# GPT wrapper -- sometimes it fails and we should retry
def gpt_wrapper(msgs, n, stop):
    while True:
        try:
            completions = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                messages= msgs,
                n = n,
                stop = stop
            )
        except Exception as e:
            if 'maximum context length' in str(e):
                print('...Error.. too long...aborting...' + str(e))
                return None
            else:
                print('...Error.. trying again...' + str(e))
        else:
            break
    return completions

def prompt_cot(num_trials, stop, input_msgs, added_msg):
    if added_msg:
        input_msgs.append({"role" : "assistant", "content" : added_msg})

    completions = gpt_wrapper(input_msgs, num_trials, stop)
    return completions

def prompt_vote(num_trials, stop, input_msgs):
    completions = gpt_wrapper(input_msgs, num_trials, stop)
    return completions

def tally_votes(gpt_votes, choices, num_trials):
    vote_results = [0] * len(choices)
    for i in range(num_trials):
        if gpt_votes.choices[i].finish_reason == 'stop':
            vote_output = gpt_votes.choices[i]['message']['content']
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(len(choices)):
                    vote_results[vote] += 1
                else:
                    print(f'INFO: GPT did not choose anything: {match}, {[vote_output]}')
            else:
                print(f'ERROR: vote no match: {[vote_output]}')
        else:
            print(f"ERROR: Voting failed for trial {i}")
    return vote_results

def solve_toc(input_msgs_cot, identify_trials, code_trials, identify_vote_trials, code_vote_trials, identify_stop, get_identified_names_func, get_identify_votes_msgs_func, get_code_votes_msgs_func):
    # Depth 1: Identify
    identify_completions = prompt_cot(num_trials = identify_trials, stop = identify_stop, input_msgs = input_msgs_cot, added_msg = None)

    # If the entire thing fails we have to return
    if identify_completions is None:
        print("ERROR: Depth 1: identifying failed")
        return None, None

    # Get identified items
    identified_names_lst = [str(item) for item in get_identified_names_func(identify_trials, identify_completions)]
    
    # Get unique options
    identified_names_set = list(set(identified_names_lst))
    identified_names_lst_set_index = [identified_names_set.index(x) for x in identified_names_lst] # so we keep track of the index in the set
        
    # if the options are the same, we return first trial
    if len(identified_names_set) == 1:
        most_popular_identify = identified_names_set[0]
        print(f"INFO: Depth 1: identified names are the same for all trials so we don't vote, {most_popular_identify}")
        if most_popular_identify == 'None':
            print("ERROR: Depth 1: identified names are None for all trials")
            return None, None
    else:
        # Depth 1: Vote on the best choice using GPT
        gpt_identify_votes = prompt_vote(
            num_trials = identify_vote_trials,
            stop = None,
            input_msgs = get_identify_votes_msgs_func(identified_names_set))
        
        # If voting fails we return here
        if gpt_identify_votes is None:
            print("ERROR: Depth 1: Voting failed")
            return None, None

        # Tally the votes
        vote_identify_results = tally_votes(gpt_votes = gpt_identify_votes, choices = identified_names_set, num_trials = identify_vote_trials)
        
        # Get the most popular choice -- if no results it will return the first one
        most_popular_identify = identified_names_set[vote_identify_results.index(max(vote_identify_results))]

    # Depth 2: Work on the code somehow
    # first identify which of the completions had the most popular choice so we get that same content
    most_popular_identify_index = identified_names_lst_set_index.index(identified_names_set.index(most_popular_identify))
    choice_msg = identify_completions.choices[most_popular_identify_index]['message']['content']
    updated_code_completions = prompt_cot(num_trials = code_trials, stop = None, input_msgs = input_msgs_cot, added_msg = choice_msg)

    # If code fails we return here
    if updated_code_completions is None:
        print("ERROR: Depth 2: code failed")
        return most_popular_identify, None

    # Get the updated code
    updated_code_lst = []
    for i in range(code_trials):
        if identify_completions.choices[i].finish_reason == 'stop':
            try:
                code = updated_code_completions.choices[i]['message']['content'].split('```')[1].strip("\n")
            except:
                print(f"ERROR: Depth 2: code failed for trial {i}")
                updated_code_lst.append(None)
            else:
                if code.startswith('python'):
                    code = code[6:]
                code = code.strip("\n")
                updated_code_lst.append(code)
        else:
            print(f"ERROR: Depth 2: code failed for trial {i}")
            updated_code_lst.append(None)
    
    # if the options are the same, we return either trial
    updated_code_set = list(set(updated_code_lst))

    if len(updated_code_set) == 1:
        if updated_code_set[0] is None:
            print("ERROR: Depth 2: updated code is None for all trials")
            return most_popular_identify, None
        print("INFO: Depth 2: updated code is the same for all trials so we don't vote")
        return most_popular_identify, updated_code_set[0]

    # Depth 2: Vote on the best choice using GPT
    gpt_code_votes = prompt_vote(
        num_trials = code_vote_trials,
        stop = None,
        input_msgs=get_code_votes_msgs_func(most_popular_identify, updated_code_set)
        )
    
    # If voting fails we return here
    if gpt_code_votes is None:
        print("ERROR: Depth 2: Voting failed")
        return most_popular_identify, None

    # Tally the votes
    vote_code_results = tally_votes(gpt_votes = gpt_code_votes, choices = updated_code_set, num_trials = code_vote_trials)
    
    # Get the most popular choice -- if no results it will return the first one
    most_popular_code = updated_code_set[vote_code_results.index(max(vote_code_results))]
    
    return most_popular_identify, most_popular_code