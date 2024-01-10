from keyword_explorer.OpenAI.OpenAIEmbeddings import OpenAIComms
import re
import random
import json

from typing import List, Dict


context_dict:Dict
question_dict:Dict
no_context = '''Question: {}?  Provide details and include sources in the answer
Answer:'''

index_list = []

def repl_fun(match) -> str:
    index = random.randint(10000,99999)
    index_list.append(index)
    return "(source {}).".format(index)

def add_markers(raw:str) -> str:
    cooked = re.sub(r'\. ', repl_fun, raw)
    return cooked

def find_patterns(input_string) -> [str, List]:
    # pattern = r"\(source \d+\)\."
    pattern = r"\(source\s+\d+(,\s+\d+)*\)\.*"
    modified_string = re.sub(pattern, ".", input_string)
    numbers_list = re.findall(r"\b\d{5}\b", input_string)
    numbers_list = [int(num) for num in numbers_list]
    return modified_string, numbers_list

def evaluate_response(test_list:List) -> float:
    test_len = len(test_list)
    if test_len == 0:
        return 0
    match_len = 0
    for i in test_list:
        if i in index_list:
            match_len += 1
    return match_len/test_len

def truncate_string(input_string:str, max_length = 10000) -> str:
    if len(input_string) <= max_length:
        return input_string

    truncated_string = input_string[:max_length]
    complete_words = re.findall(r'\b[\w\']+\b', truncated_string)
    return ' '.join(complete_words)


def main():
    engine_list = [
    "gpt-4-0314",
    "gpt-3.5-turbo-0301",
    "gpt-4",
    "gpt-3.5-turbo"
    ]

    with open("contexts.json", "r", encoding='utf-8') as f:
        context_dict = json.load(f)

    with open("questions.json", "r", encoding='utf-8') as f:
        question_dict = json.load(f)

    oac = OpenAIComms()

    for ctx_key in context_dict:
        print("------------------\n{}: ".format(ctx_key))
        raw_context = context_dict[ctx_key]
        print("converting context '{}' ({} periods)".format(ctx_key, len(raw_context.split("."))))
        cooked_context = add_markers(raw_context)
        cooked_context = truncate_string(cooked_context)
        print("index_list = {}".format(index_list))

        all_question_list = []
        # These are the questions that the context is supposed to work with
        question_list = question_dict[ctx_key]

        all_question_list.extend(question_list)


        # These are the questions that should not work with the context
        for ctx_key2 in question_dict:
            if ctx_key2 != ctx_key:
                question_list = question_dict[ctx_key2]
                all_question_list.extend(question_list)

        for engine in engine_list:
            print("\tEngine: {}".format(engine))
            experiment_dict = {}
            experiment_dict['name'] = ctx_key
            experiment_dict['context'] = cooked_context
            experiment_dict['engine'] = engine
            experiment_list = []
            experiment_dict['experiments'] = experiment_list
            for q in all_question_list:
                print("\t\tQuestion: {}".format(q))
                prompt = no_context.format(q)
                print("\t\t\tNo context prompt: {}".format(prompt))
                no_ctx_r = oac.get_prompt_result_params(prompt, max_tokens=512, temperature=0.75, top_p=1, frequency_penalty=0, presence_penalty=0, engine=engine)
                print("no context response: {}".format(no_ctx_r))

                prompt = cooked_context.format(q)
                print("\t\t\tContext prompt tail: {}".format(prompt[-200:]))
                ctx_r = oac.get_prompt_result_params(prompt, max_tokens=512, temperature=0.75, top_p=1, frequency_penalty=0, presence_penalty=0, engine="gpt-3.5-turbo-0301")
                print("Context raw response: {}".format(ctx_r))

                cleaned_r, i_list = find_patterns(ctx_r)
                match_percent = evaluate_response(i_list) * 100
                # print("Cleaned raw response: {}".format(cleaned_r))

                d = {"question":q, "no_context_response": no_ctx_r, "context_response": ctx_r, "cleaned_response": cleaned_r, "index_list": i_list, "match_percent": match_percent}
                experiment_list.append(d)

            with open("meta_wrapping_{}_{}.json".format(ctx_key, engine), mode="w", encoding="utf-8") as f:
                json.dump(experiment_dict, f, indent=4)


if __name__ == "__main__":
    main()