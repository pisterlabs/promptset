from configuration import ITERATIONS, PREPROMPT_P0, PREPROMPT_SEM, PREPROMPT_PSY, PREPROMPT_CONTXT, PREPROMPT_SENTENCE_TO_INPUT
from configuration import TEST_FILES_LIST as files_list
from openai_queries import experiment
from secret_stuff import API_KEYS
from json import load, dump
from Children import Children
from functions import calculate_score

prompts = []

prompts.append(["PREPROMPT_P0", PREPROMPT_P0 + PREPROMPT_SENTENCE_TO_INPUT])
#prompts.append(["PREPROMPT_SEM", PREPROMPT_SEM + PREPROMPT_SENTENCE_TO_INPUT])
#prompts.append(["PREPROMPT_PSY", PREPROMPT_PSY + PREPROMPT_SENTENCE_TO_INPUT])
#prompts.append(["PREPROMPT_CONTXT", PREPROMPT_CONTXT + PREPROMPT_SENTENCE_TO_INPUT])
prompts.append(["PREPROMPT_P0_SEM", PREPROMPT_P0 + PREPROMPT_SEM + PREPROMPT_SENTENCE_TO_INPUT])
prompts.append(["PREPROMPT_P0_PSY", PREPROMPT_P0 + PREPROMPT_PSY + PREPROMPT_SENTENCE_TO_INPUT])
prompts.append(["PREPROMPT_P0_CONTXT", PREPROMPT_CONTXT + PREPROMPT_P0 + PREPROMPT_SENTENCE_TO_INPUT])
prompts.append(["PREPROMPT_P0_CONTXT_SEM", PREPROMPT_CONTXT + PREPROMPT_P0 + PREPROMPT_SEM + PREPROMPT_SENTENCE_TO_INPUT])
prompts.append(["PREPROMPT_P0_CONTXT_PSY", PREPROMPT_CONTXT + PREPROMPT_P0 + PREPROMPT_PSY + PREPROMPT_SENTENCE_TO_INPUT])
prompts.append(["PREPROMPT_P0_SEM_PSY", PREPROMPT_P0 + PREPROMPT_SEM + PREPROMPT_PSY + PREPROMPT_SENTENCE_TO_INPUT])
prompts.append(["PREPROMPT_P0_CONTXT_SEM_PSY", PREPROMPT_CONTXT + PREPROMPT_P0 + PREPROMPT_SEM + PREPROMPT_PSY + PREPROMPT_SENTENCE_TO_INPUT])


for index, prompt in enumerate(prompts):
    children = []
    results = {
        "prompt": "",
        "results": [],
    }
    print(f"Starting test {index + 1} with prompt {prompt[1]}")
    for index, file in enumerate(files_list):
        with open(file, "r", encoding="utf-8") as json_file:
            data_list = load(json_file)
        children.append([file, Children(target=experiment, args=(data_list, API_KEYS[index], prompt[1], ITERATIONS))])
        children[index][1].start()
        print(f'Thread {index + 1} started')

    for index, child in enumerate(children):
        results["prompt"] = prompt[1]
        results["results"].append([children[index][0], children[index][1].join()])

    print(results)

    with open(f"test_results/performance_results_{prompt[0]}.json", "w") as file:
        dump(results, file, indent=4)