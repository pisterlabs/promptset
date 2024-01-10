import os
import json

import re
import openai
import math

from radon.complexity import cc_visit
from dotenv import load_dotenv
load_dotenv()

N = 10
openai.api_key = os.environ["OPENAI_API_KEY"]

env_file = os.getenv("updatedFiles")
old_and_new_code = json.loads(env_file)
print("ENV:", old_and_new_code)


def call_Chat_gpt_for_time_and_space_complexity(content):
    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
              "role": "system",
              "content": "You will be provided with Python code, give only Time complexity and Space Complexity all functions in json fomat with no explation"
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return chat_response['choices'][0]['message']['content']


def get_cyclomitic_complexity(fun):
    return cc_visit(fun)


def convert_complexity_to_number(complexity):
    final_comp = 1
    complexity = complexity[2:-1]
    log_indexes = [i.start() for i in re.finditer('log', complexity)]
    complexity = complexity.replace('log', '')

    complexity = re.sub(r'[a-zA-Z]', r'n', complexity)
    for id in log_indexes:
        complexity = complexity[:id+1]+"log"+complexity[id+1:]
    complexity = complexity.replace('  ', '')

    complexity = list(complexity)

    i = 0
    while i < len(complexity):
        if complexity[i] == "n":
            final_comp *= N
        elif complexity[i] == "l":
            final_comp *= 1.2
            i += 3
        elif complexity[i] == "^":
            last = complexity[i-1]
            if last.isnumeric():
                last = int(last)
            else:
                last = N
            next = int(complexity[i+1]) if complexity[i+1].isnumeric() else N

            if final_comp > 1:
                final_comp /= last
            if next > last:
                final_comp = final_comp * 100  # math.pow(last,next)
            elif next == last:
                final_comp = final_comp * 150
            else:
                final_comp = final_comp * 70
            i += 1
        i += 1
    return final_comp
#   if


def give_start_rating(old_score, new_score):
    delta = ((old_score-new_score)/old_score)*100
    if delta <= 0:
        print("No Optimisation Required")
        return {'old_code': 4.5,
                'new_code': 4.5}

    else:
        if 0 < delta <= 20:
            return {'old_code': 4,
                    'new_code': 4.5}
        elif 20 < delta <= 50:
            return {'old_code': 3,
                    'new_code': 4.5}
        elif 50 < delta <= 75:
            return {'old_code': 2.5,
                    'new_code': 4.5}
        else:
            return {'old_code': 1.5,
                    'new_code': 4.5}


def get_score_for_code(fun):

    print("Calling ChatGPT API to Get Complexities")
    resp = call_Chat_gpt_for_time_and_space_complexity(fun)
    resp = json.loads(resp)

    print("Getting Cyclomatic Complexity")
    cyclo_comp = get_cyclomitic_complexity(fun=fun)

    for c in cyclo_comp:
        name, comp = c.name, c.complexity
        resp[name]["cyclo_complexity"] = comp

    for res in resp:
        code = resp[res]
        score = convert_complexity_to_number(
            code["time_complexity"])+convert_complexity_to_number(code["space_complexity"])+code["cyclo_complexity"]
        resp[res]["score"] = score

    return resp

def normlise_scores(old_scores,new_scores):
    max_score = max(old_scores+new_scores)
    min_score = min(old_scores+new_scores)
    print("Max Score:", max_score)
    print("Min Score:", min_score)

    normalise_old_score = 0
    for old_score in old_scores:
        normalise_old_score += (old_score - min_score)/(max_score - min_score) 
    normalise_new_score = 0
    for new_score in new_scores:
        normalise_new_score += (new_score - min_score)/(max_score - min_score)
    
    return (normalise_old_score,normalise_new_score)


if __name__ == "__main__":

    star_rating_dict = {}
    old_pr_score = 0
    new_pr_score = 0
    new_scores_list = []
    old_scores_list = []
    for codes in old_and_new_code:
        new_code = codes["newCode"]
        old_code = codes["oldCode"]

        print("unoptimised Code")
        path = 'utils/unoptimised_code.py'
        score_resp_unoptimised = get_score_for_code(old_code)

        print("\n\noptimised Code")
        path = 'utils/optimised_code.py'
        score_resp_optimised = get_score_for_code(new_code)

        star_rating_dict[codes['path']] = []

        print("\n\n")
        for function in score_resp_unoptimised:
            print(f"Calculating Score for Function {function}")
            old_score, new_score = score_resp_unoptimised[function]["score"], score_resp_optimised[function]["score"]
            if old_score < new_score:
                new_score = old_score
            old_pr_score += old_score
            new_pr_score += new_score
            old_scores_list.append(old_score)
            new_scores_list.append(new_score)
            print(f"Score for unoptimised Function {old_score}")
            print(f"Score for optimised Function {new_score}")
            print(f"Calculating Sart Rating for Function {function}")
            star_rating = give_start_rating(old_score, new_score)

            old_star, new_star = star_rating["old_code"], star_rating["new_code"]
            old_extra = 0 if math.ceil(old_star) == old_star else 1
            new_extra = 0 if math.ceil(new_star) == new_star else 1

            old_star_rating = "\u2B50" * math.floor(old_star)+"\u2605"*old_extra
            new_star_rating = "\u2B50" * math.floor(new_star)+"\u2605"*new_extra
            print("Old Code Star Rating: "+old_star_rating)
            print("New Code Star Rating:"+new_star_rating)
            print("\n\n")

            function_rating_dict = {
                function: {
                    'old_score': old_score,
                    'new_score': new_score,
                    'old_star_rating': old_star_rating,
                    'new_star_rating': new_star_rating
                }
            }
            star_rating_dict[codes['path']].append(function_rating_dict)

    normalise_old_pr_score, normalise_new_pr_score = normlise_scores(old_scores_list, new_scores_list)

    print(f'star rating dict {json.dumps(star_rating_dict)}')

    star_rating = give_start_rating(normalise_old_pr_score, normalise_new_pr_score)

    old_pr_star, new_pr_star = star_rating["old_code"], star_rating["new_code"]
    old_extra = 0 if math.ceil(old_star) == old_star else 1
    new_extra = 0 if math.ceil(new_star) == new_star else 1

    old_pr_star_rating = "\u2B50" * math.floor(old_pr_star)+"\u2605"*old_extra
    new_pr_star_rating = "\u2B50" * math.floor(new_pr_star)+"\u2605"*new_extra
    print(f"old pr score: {old_pr_score}")
    print(f"new pr score: {new_pr_score}")
    print("Old Code Star Rating old_pr_star_rating: "+old_pr_star_rating)
    print("New Code Star Rating new_pr_star_rating:"+new_pr_star_rating)

    env_file = os.getenv('GITHUB_ENV')

    with open(env_file, "a") as myfile:
        myfile.write(f"star_ratings={json.dumps(star_rating_dict)}\n")
        myfile.write(f"old_pr_score={old_pr_score}\n")
        myfile.write(f"new_pr_score={new_pr_score}\n")
        myfile.write(f"old_pr_star_rating={old_pr_star_rating}\n")
        myfile.write(f"new_pr_star_rating={new_pr_star_rating}\n")
