import openai
import pandas as pd
import multiprocessing
import time
import logging
from multiprocessing import Pool

# Your API key
openai.api_key = 'your-api-key'  

#assign experts for target
target_role_map ={}

#for example
# target_role_map ={ 
#     "Climate Change is a Real Concern": "environmental scientist",
#     "Feminist Movement": "sociologist",
#     "Hillary Clinton": "political scientist"
# }

def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")

def get_completion_with_role(role, instruction, content):
    messages = [
        {"role": "system", "content": f"You are a {role}."},
        {"role": "user", "content": f"{instruction}\n{content}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message["content"]

def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message["content"]

#you may adjust these functions according to your text and target of interest.
def linguist_analysis(tweet):
    instruction = "Accurately and concisely explain the linguistic elements in the sentence and how these elements affect meaning, including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on. Do nothing else."
    return get_completion_with_role("linguist", instruction, tweet)

def expert_analysis(tweet, target):
    role = target_role_map.get(target, "expert")
    instruction = f"Accurately and concisely explain the key elements contained in the quote, such as characters, events, parties, religions, etc. Also explain their relationship with {target} (if exist). Do nothing else."
    return get_completion_with_role(role, instruction, tweet)

def user_analysis(tweet):
    instruction = "Analyze the following sentence, focusing on the content, hashtags, Internet slang and colloquialisms, emotional tone, implied meaning, and so on. Do nothing else."
    return get_completion_with_role("heavy social media user", instruction, tweet)

def stance_analysis(tweet, ling_response, expert_response, user_response, target, stance):
    role = target_role_map.get(target, "expert")
    return get_completion(f"'''{tweet}'''\n <<<{ling_response}>>>\n [[[{expert_response}]]]\n---{user_response}---\n\
                          You think the attitude behind the sentence surrounded by ''' ''' is {stance} of {target}. \
                          The content enclosed by <<< >>> represents linguistic analysis. The content within [[[ ]]] represents the analysis of a {role}. \
                          The content enclosed by --- ---  represents the analysis of a heavy social media user. Identify the top three pieces of evidence from these that best support your opinion and argue for your opinion.")

def final_judgement(tweet, favor_response, against_response, target):
    judgement=get_completion(f"Determine whether the sentence is in favor of or against {target}, or is irrelevant to {target}.\n \
                             Sentence: {tweet}\nJudge this in relation to the following arguments:\n\
                                Arguments that the attitude is in favor: {favor_response}\n\
                                    Arguments that the attitude is against: {against_response}\n\
                                            Choose from:\n A: Against\nB: Favor\nC: Irrelevant\n Constraint: Answer with only the option above that is most accurate and nothing else.")
    print(judgement)
    return judgement
    

def worker_process(chunk_tuple):
    index, row = chunk_tuple
    tweet = row['Tweet']
    target = row['Target']

    # Step 1: Linguist analysis
    ling_response = linguist_analysis(tweet)

    # Step 2: Expert analysis
    expert_response = expert_analysis(tweet, target)

    # Step 3: Heavy social media user analysis
    user_response = user_analysis(tweet)

    # Step 4: Debate
    favor_response = stance_analysis(tweet, ling_response, expert_response, user_response, target, "in favor")
    against_response = stance_analysis(tweet, ling_response, expert_response, user_response, target, "against")

    # Step 5: Final judgement
    final_response = final_judgement(tweet, favor_response, against_response, target)

    result = {
        'index': index, 
        'Tweet': tweet,
        'Target': target,
        'Linguist Analysis': ling_response,
        'Expert Analysis': expert_response,
        'User Analysis': user_response,
        'In Favor': favor_response,
        'Against': against_response,
        'Stance': final_response
    }

    return result


def add_predictions(data):
    num_cores = 50
    pool = multiprocessing.Pool(processes=num_cores)

    results = list(pool.imap_unordered(worker_process, data.iterrows()))
  
    results_sorted = sorted(results, key=lambda x: x['index'])

    for res in results_sorted:
        idx = res.pop('index')
        for key, value in res.items():
            data.at[idx, key] = value

    pool.close()
    pool.join()

data = load_csv_data("your_data.csv")
add_predictions(data)
data.to_csv("results.csv", index=False)
