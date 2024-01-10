import openai
import json
import os
import datetime
from time import sleep

LIMIT_POSTS = 2000
LIMIT_API_REQUESTS = 20000
SKIP_ROUND_0 = True
START_AT_POST = 0
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LIMIT_WORDS_MIN = 100  # min number of words per request (soft constraint)
LIMIT_WORDS_MAX = 450  # max number of words per req
LIMIT_CHARACTERS_REQUEST = 3900
SYSTEM_PROMPT = "You summarize Reddit posts about ChatGPT in 20-50 words. If you don't know, you reply with 'NA'."
SYSTEM_PROMPT_SUMMARIES = "You analyze summaries (made by chatgpt) of Reddit posts and say the most relevant points " \
                          "in 50-80  words. You can combine or omit summaries if relevant. Don't explain what ChatGPT " \
                          "is. Give a higher priority to the opinion of the users. I'm interested in finding out how " \
                          "users feel about ChatGPT "
TIMER = 1  # Waiting time in seconds between each request to avoid being blocked by server
POSTS_DIR = '../data/posts'
OUTPUT_DIR = '../data/output'


def ask_chatgpt(system_prompt, user_prompt):
    total_tokens = len(system_prompt) + len(user_prompt)
    user_prompt_max = LIMIT_CHARACTERS_REQUEST - len(system_prompt)
    if total_tokens > LIMIT_CHARACTERS_REQUEST:
        user_prompt = user_prompt[0:user_prompt_max] + ' [post is too long to display full]'
    openai.api_key = OPENAI_API_KEY
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.7)
    reply = chat.choices[0].message.content
    return reply


def get_post_prompt(post):
    prompt = ""
    prompt += "\n Subreddit: " + post['subreddit']
    prompt += "\nTitle: " + post['title']
    # prompt += "\nIs_Self: " + str(post['is_self'])
    prompt += "\nURL: " + str(post['url'])
    prompt += "\nContent: " + post['content']
    return prompt

def get_group_prompt(group):
    prompt = ""
    for summary in group:
        prompt += summary
        prompt += "\n"
    prompt = prompt.rstrip('\n')
    with open('test.json', 'w') as f:
        prompt_summary_dict = {'prompt': prompt}
        json.dump(prompt_summary_dict, f)
    return prompt


def count_words(string):
    words = string.split()
    return len(words)


def get_post_prompts_from_files(posts_dir, limit_posts):
    prompts = []
    filenames = [filename for filename in os.listdir(posts_dir)]
    filenames.sort()
    for filename in filenames:
        if filename.endswith('.json'):
            print(filename)
            with open(os.path.join(posts_dir, filename), 'r') as file:
                posts = json.load(file)
                prompts += [get_post_prompt(post) for post in posts]
        if len(prompts) >= limit_posts:
            prompts = prompts[0:limit_posts]
            break
    # Remove Duplicates:
    unique_prompts = []
    for prompt in prompts:
        if prompt not in unique_prompts:
            unique_prompts.append(prompt)
    return unique_prompts


def group_strings_by_word_count(strings, min_word_count, target_word_count):
    groups = []
    current_group = []
    current_word_count = 0
    target_word_count = 500
    min_word_count = 100

    for string in strings:
        string_word_count = count_words(string)

        # if adding the current string to the current group would exceed the maximum or fall below the minimum word count, start a new group
        if (current_word_count + string_word_count > target_word_count and current_word_count >= min_word_count) or current_word_count + string_word_count > target_word_count + min_word_count:
            groups.append(current_group)
            current_group = []
            current_word_count = 0

        # add the current string to the current group
        current_group.append(string)
        current_word_count += string_word_count

    # add any remaining strings to the last group
    if current_group:
        groups.append(current_group)

    return groups


start_time = datetime.datetime.now()

# Round 0: Summarize individual reddit posts
nr_of_api_requests = 0
if not SKIP_ROUND_0:
    print("ROUND 0:")
    prompts = get_post_prompts_from_files(POSTS_DIR, LIMIT_POSTS)
    # print(len(prompts))
    # print(sum([count_words(prompt) for prompt in prompts]))
    for i in range(0, len(prompts)):
        if i >= START_AT_POST:
            prompt = prompts[i]
            print(str(i + 1) + "/" + str(len(prompts)))
            if nr_of_api_requests >= LIMIT_API_REQUESTS:
                Print("LIMIT")
            sleep(TIMER)
            prompt_summary = ask_chatgpt(SYSTEM_PROMPT, prompt)
            nr_of_api_requests += 1
            with open(f'{OUTPUT_DIR}/summaries_0_{str(i)}.json', 'w') as f:
                prompt_summary_dict = {'prompt': prompt, 'summary': prompt_summary}
                json.dump(prompt_summary_dict, f)

# Rounds 1,2...: Summarize summaries:
last_round = False
round_number = 0
while not last_round:
    round_number +=1
    print("ROUND " + str(round_number))
    summaries_before = []
    filenames = [filename for filename in os.listdir(OUTPUT_DIR)]
    filenames.sort()
    for filename in filenames:
        if filename.startswith('summaries_'+str(round_number-1)):
            with open(os.path.join(OUTPUT_DIR, filename), 'r') as file:
                data = json.load(file)
                summaries_before.append(data['summary'])
    groups = group_strings_by_word_count(summaries_before, LIMIT_WORDS_MIN, LIMIT_WORDS_MAX)
    if len(groups) == 1:
        last_round = True
    prompts = [get_group_prompt(group) for group in groups]
    for i in range(0, len(prompts)):
        prompt = prompts[i]
        print(str(i + 1) + "/" + str(len(prompts)))
        prompt_summary = ask_chatgpt(SYSTEM_PROMPT_SUMMARIES, prompt)
        nr_of_api_requests += 1
        with open(f'{OUTPUT_DIR}/summaries_{str(round_number)}_{str(i)}.json', 'w') as f:
            prompt_summary_dict = {'prompt': prompt, 'summary': prompt_summary}
            json.dump(prompt_summary_dict, f)

elapsed_time = datetime.datetime.now() - start_time
print('Finished summarizing posts.')
print(f'Total number of API requests: {str(nr_of_api_requests)}')
print(f"Time elapsed: {elapsed_time}")
