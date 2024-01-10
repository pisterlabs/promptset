from binhex import openrsrc
from curses import init_pair
import os
from unittest import result
import openai
import json
import argparse
# from similarity import compute_cosine_similarity

openai.api_key = os.getenv("OPENAI_API_KEY")
example = {
        "puzzle": ["A man walks into a bar and asks the bartender for a drink of water.",
                    "The bartender pulls out a qun, points it at the man and cocks it. ",
                    "The man pauses before saying \"Thank you\"and leaving.",
                    "What happened?"
        ],
        "truth": ["The man had the hiccups."
                  "The bartender realized this and chose instead to cure the hiccups by frightening the man with the gun."
        ],
        "follow-up-Q":["Could the bartender hear him?",
                       "Did the man ask for water in an offensive way?"],
        "follow-up-A":["Yes.","No."],
        "hint":"As we said, there is nothing wrong with either the building or the elevator. There is, however, some feature of the man that causes him to take the stairs from the seventh floor."
    }

def load_data(data_path):
	with open(data_path, 'r') as f:
		data_set = json.load(f)
	return data_set

def extract_input_item(data):
    puzzle = "".join(data["puzzle"])
    truth = "".join(data["truth"])
    fol_q = list(data["follow-up-Q"])
    fol_a = list(data["follow-up-A"])
    return puzzle, truth, fol_q, fol_a

def get_response(prompt):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n\n"]
        )
    output_ai = response.get('choices')[0]['text'].split("\n")[0]
    if output_ai == "":
        print("***** connect failure !! try again please.*****")
    return output_ai

def question_generation_input(data_item, THINK_HINT):
    # task_description = "I am an intelligent bot that can play situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details."
    task_description = "I can generate five questions based on the \"puzzle\" that start with \"Do/Am/Is/Are/Can/Could/Does/Did/Was/Were\" and can only be answered with \"yes\" or \"no\"."
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    example_prefix = "puzzle:" + "".join(example["puzzle"]) + "\n"
    example_prefix += "follow_up_Q:"+example["follow-up-Q"][0]+"\n"+"follow_up_A:"+example["follow-up-A"][0] + "\n"
    example_prefix += "follow_up_Q:"+example["follow-up-Q"][1] + "\n"
    input_prefix = "puzzle:" + puzzle + "\n"  
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_up_Q:"+ fq
            input_prefix += "follow_up_A:"+ fa
        input_prefix += "\n"
    input_prefix += THINK_HINT + "follow_up_Q:"
    final_prefix = task_description + '\n' + example_prefix + input_prefix
    print("---------- question generation ----------")
    # print(final_prefix)
    return final_prefix

def answer_generation_input(data_item):
    # task_description = "I am an intelligent bot that can play as judge in situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details, and I will give \"Yes/No/Irrelevent\" as answer to questions."
    task_description = "I can answer the question with only yes or no or irrelevant with the \"truth\". "
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    example_prefix = "truth:" + "".join(example["truth"]) + "\n"
    example_prefix += "follow_up_Q:"+example["follow-up-Q"][0]+ "\n" +"follow_up_A:"+example["follow-up-A"][0] + "\n"
    example_prefix += "follow_up_Q:"+example["follow-up-Q"][1]+ "\n" +"follow_up_A:"+example["follow-up-A"][1] + "\n"
    input_prefix = "truth:" + truth + "\n"  
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_up_Q:"+ fq
            input_prefix += "follow_up_A:"+ fa
        # input_prefix += "\n"
    input_prefix += "\n"+"follow_up_Q:" + data_item["follow-up-Q"][-1] 
    input_prefix += "\n" + "follow_up_A:"
    final_prefix = task_description + '\n' + example_prefix + input_prefix
    print("---------- answer generation ----------")
    # print(final_prefix)
    return final_prefix

def solution_generate_input(data_item):
    # task_description = "I am an intelligent bot that can play situation puzzles with user. A puzzle is given first, and the user begin to ask \"yes/no\" question to ensure details, then I will give the question a\"yes/no/irrelevent\" answer. Finally user try to give solution for the puzzle."
    task_description = "I can generate a logical answer to the question based on the puzzle, the follow_up_Q and the follow_up_Q. "
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    example_prefix = "puzzle:" + "".join(example["puzzle"]) + "\n"
    example_prefix += "follow_up_Q:"+example["follow-up-Q"][0]+"\n"+"follow_up_A:"+example["follow-up-A"][0] + "\n"
    example_prefix += "solution:" + "".join(example["truth"]) + "\n"
    input_prefix = "puzzle:" + puzzle + "\n"  
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_up_Q:"+ fq
            input_prefix += "follow_up_A:"+ fa
        # input_prefix += "\n"
    input_prefix += "\n" + "solution:"
    final_prefix = task_description + '\n' + example_prefix + input_prefix
    print("---------- solution generation ----------")
    # print(final_prefix)
    return final_prefix

if __name__=="__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--sample_num', type=int, default=1)
    parse.add_argument('--chance_num', type=int, default=3)
    parse.add_argument('--threshold', type=int, default=0.8)
    # parse.add_argument('--with_hint', action='store_true')
    args = parse.parse_args()
    dataset = load_data("./example.json")
    dataset = list(dataset.values())
    output_file = "./output.txt"
    output_dataset = "./new_dataset.json"
    thinking_hint = ["correctly think!","reverse thinking!","think differently!",""]
    think_hint = ""
    # print(extract_input_item(dataset[0]))
    # print(question_generation_prompt(dataset[0]))
    # print(answer_generation_prompt(dataset[0]))
    # print(solution_generate_prompt(dataset[0]))
    for ditem in dataset[:args.sample_num]:
        # print(data_item)
        chance_count = 0
        while chance_count < args.chance_num:
            # genrate question
            print("######## follow-up-A #####",ditem["follow-up-A"])
            if ditem["follow-up-A"]:
                if ditem["follow-up-A"][-1] == "Yes.":
                    think_hint = thinking_hint[0]
                elif ditem["follow-up-A"][-1] == "No.":
                    think_hint = thinking_hint[1] 
                elif ditem["follow-up-A"][-1] == "irrelevant.":
                    think_hint = thinking_hint[2]
                elif ditem["follow-up-A"][-1] not in thinking_hint:
                    print("The current answer is not in the correct format!!! ")
                    chance_count += 1
                    continue
            question_generation_prompt = question_generation_input(ditem,THINK_HINT=think_hint)
            print("**************")
            print(question_generation_prompt)
            print("*************")
            generated_question = get_response(question_generation_prompt)
            ditem["follow-up-Q"].append(generated_question)
            print("*",generated_question)
            # generate answer
            if generated_question =="":
                chance_count += 1
                continue
            answer_generation_prompt = answer_generation_input(ditem)
            generated_answer = get_response(answer_generation_prompt)
            ditem["follow-up-A"].append(generated_answer)
            print("*",generated_answer)
            chance_count += 1
        # generate solution
        solution_generate_prompt = solution_generate_input(ditem)
        generated_solution = get_response(solution_generate_prompt)
        print("*",generated_solution)
            # print(f"the {chance_count+1} chance's answer:", generated_solution)
        
            # calculate the current solution's accuracy
        #     similarity_score = compute_cosine_similarity("".join(data_item["truth"]), generated_solution)
        #     if similarity_score >= args.threshold:
        #         break
        # with open(output_file,"w",encoding="utf-8") as fp:
        #     fp.write(generated_solution)
        with open(output_dataset,"w",encoding="utf-8") as fd:
            json.dump(dataset[:args.sample_num],fd)