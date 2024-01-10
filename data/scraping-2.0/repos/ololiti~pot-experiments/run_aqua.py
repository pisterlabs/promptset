import json
from openai import OpenAI
import argparse
import func_timeout
import math
from tqdm import tqdm
import time
from collections import Counter
import sympy
from sympy.solvers import solve
from sympy import Symbol, Eq
from sympy import simplify



parser = argparse.ArgumentParser()
parser.add_argument("--name", default="test", type=str)
parser.add_argument("--data_file", default="aqua_test.jsonl", type=str)
parser.add_argument("--num_examples", default=5, type=int)
parser.add_argument("--verbose", default=False, type=bool)
parser.add_argument("--system_prompt", default="aqua_system.txt", type=str)
parser.add_argument("--example_prompt", default="gpt4-aqua-fewshot1.txt", type=str)
parser.add_argument("--mode", default="normal", type=str)
parser.add_argument("--teacher_prompt", default="gpt3-teacher-prompt2.txt", type=str)
parser.add_argument("--mistake_prompt", default="gpt4-mistake-prompt.txt", type=str)
parser.add_argument("--best_of", default=1, type=int)

args = parser.parse_args()

def load_data(filename):
    test_data = []
    with open(f"data/{filename}") as f:
        for line in f:
            tmp = json.loads(line)
            test_data.append(tmp)
    return test_data

def load_prompt(sys_filename, main_filename):
    with open(f"prompts/{sys_filename}", 'r') as f:
        sys_prompt = f.read()
    with open(f"prompts/{main_filename}", 'r') as f:
        ex_prompt = f.read()
    return sys_prompt, ex_prompt

def generate_example(ex_prompt, example):
    example_string = f"# Question: {example['question']}\n# Answer Options: {example['options']}\n\n# Solution: \n"
    return ex_prompt + "\n\n" + example_string

def ask_gpt(system_prompt, user_prompt, model="gpt-4", max_tokens=None):
    client = OpenAI(api_key=API_KEY)
    got_result = False
    while not got_result:
        try:
            if max_tokens is not None:
                result = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                                {"role": "user", "content":  user_prompt}],
                    max_tokens=max_tokens
                )
            else:
                result = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                                {"role": "user", "content":  user_prompt}]
                )
            got_result = True
        except Exception as e:
            print(e)
            time.sleep(3)
    result = result.choices[0].message.content
    return result

def pair_student(system_prompt, user_prompt, question, verbose):
    result1 = ask_gpt(system_prompt, user_prompt)
    result2 = ask_gpt(system_prompt, user_prompt)
    answer1 = safe_execute(result1, verbose)
    answer2 = safe_execute(result2, verbose)
    if verbose:
        print(f"{result1}\n{result2}\nanswer 1: {answer1}, answer2: {answer2}")
    if abs(answer1 - answer2) < 0.001:
        return result1, answer1
    system_prompt2 = system_prompt + "\n# Two students have attempted to write code to solve this math problem, but at least one of them has reasoned about the problem incorrectly."
    user_prompt2 = f"# Question: {question} \n\n# Student 1 Solution (possibly incorrect): \n{result1}\n# This gives an answer of {answer1}. \n\n# Student 2 Solution (possibly incorrect):\n{result1}\n# This gives an answer of {answer2}.\n\n# Question: {question} \n\n# Correct Solution: \n"
    result = ask_gpt(system_prompt2, user_prompt2)
    if verbose:
        print("Final Solution: ")
        print(result)
    executed_ans = safe_execute(result, verbose)
    return result, executed_ans

def ask_mistake(mistake_file, question, answer, verbose):
    with open(f"prompts/{mistake_file}", 'r') as f:
        sys_prompt = f.read()
    ex_prompt = "# Question: " + question + "\n\n" + "# Student Solution: \n" + answer
    result = ask_gpt(sys_prompt, ex_prompt, "gpt-4")
    if verbose:
        print(f"\n{result}")
    executed_ans = safe_execute(result, verbose)
    if executed_ans is not None:
        return result
    return None

def ask_teacher(teacher_prompt_file, question, answer):
    with open(f"prompts/{teacher_prompt_file}", 'r') as f:
        sys_prompt = f.read()
    ex_prompt = "# Question: " + question + "\n\n" + "# Student Solution: \n" + answer
    return ask_gpt(sys_prompt, ex_prompt, "gpt-4", 100)

def ask_student(system_prompt, user_prompt, orig_answer, teacher_feedback):
    teacher_response = "# A teacher has given you the following feedback on your answer. Please rewrite your Python code based on this feedback. \n # " + teacher_feedback
    teacher_response += "\n \n # Solution: \n"
    client = OpenAI(api_key=API_KEY)
    got_result = False
    while not got_result:
        try:
            result = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content":  user_prompt},
                            {"role": "assistant", "content": orig_answer},
                            {"role": "user", "content": teacher_response}]
            )
            got_result = True
        except Exception as e:
            print(e)
            time.sleep(3)
    result = result.choices[0].message.content
    return result

# prompt modified from Program of Thoughs paper
def select_option(question, options, prediction):
    sys_prompt = "Indicate the letter of the option that most closely matches the prediction."
    prompt = """
Question: A company produces 420 units of a particular computer component every month, at a production cost to the company of $110 per component, and sells all of the components by the end of each month. What is the minimum selling price per component that will guarantee that the yearly profit (revenue from sales minus production costs) will be at least $626,400 ?
Options: ['A)226', 'B)230', 'C)240', 'D)260', 'E)280']
Prediction: 234.28571428571428
Closest Option: B

Question: In how many ways can the letters of the word "PROBLEC" be rearranged to make 7 letter words such that none of the letters repeat?
Options: ['A)2!', 'B)3!', 'C)7!', 'D)8!', 'E)9!']
Prediction: 5040
Closest Option: C

Question: An exam is given in a certain class. The average (arithmetic mean) of the highest score and the lowest score is equal to x. If the average score for the entire class is equal to y and there are z students in the class, where z > 5, then in terms of x, y, and z, what is the average score for the class excluding the highest and lowest scorers?
Options: ['A)(zy – 2x)/z', 'B)(zy – 2)/z', 'C)(zx – y)/(z – 2)', 'D)(zy – 2x)/(z -2)', 'E)(zy – x)/(z + 2)']
Prediction: (-2*x + y*z)/(z - 2)
Closest Option: D

Question: Find the total no. of distinct bike no.'s that can beformed using 2 letters followed by 2 no.'s. How many letters need to be distinct?
Options: ["A)74453", "B)64543", "C)74325", "D)65000", "E)97656"]
Prediction = 67600
Closest Option: D

Question: A wire in the shape of rectangle of length 27 cm and breadth 17 cm is rebent to form a square. What will be the measure of each side?
Options: ['A)9', 'B)11', 'C)22', 'D)25', 'E)31']
Prediction = [-21.42428528562855, 21.42428528562855]
Closest Option: C

Question: A point on the edge of a fan blade that is rotating in a plane 10 centimeters from the center of the fan. What is the distance traveled, in centimeters, by this point after 30 seconds when the fan runs at the rate of 300 revolutions per minutes?
Options: ['A)750pi', 'B)1500pi', 'C)1875pi', 'D)3000pi', 'E)7500pi']
Prediction: 9424.77
Closest Option: D
    """
    prompt += f'\nQuestion: {question}\nOptions: {options}\nPrediction: {prediction}\nClosest Option: '
    response = ask_gpt(sys_prompt, prompt, max_tokens=1)
    return response


# taken from the Program of Thought paper
def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans

# safe execute function taken from Program of Thought paper
def safe_execute(code_string: str, verbose=False, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                return locals_.get('ans', None)
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception as e:
            if verbose:
                print(e)
            return None
    try:
        ans = func_timeout.func_timeout(5, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return floatify_ans(ans)

def rerun(system_prompt, user_prompt, n, verbose):
    answers = Counter()
    for i in range(n):
        result = ask_gpt(system_prompt, user_prompt)
        if verbose:
            print(result)
        executed_ans = safe_execute(result, verbose)
        if executed_ans is not None:
            answers.update([executed_ans])
    if len(answers) == 0:
        return result, None
    return result, answers.most_common(1)[0][0]

def run_test(verbose, name, test_data, prompts, reruns, mode, **kwargs):
    total = len(test_data)
    correct = 0
    non_null = 0
    system_prompt = prompts[0]
    ex_prompt = prompts[1]
    examples_in_ex_prompt = 2

    print(f"num examples: {len(test_data)}")
    output_file = f"outputs/{name}.json"

    writer = open(output_file, 'a')
    writer.write(json.dumps({"system prompt": system_prompt, "few shot prompt": ex_prompt}) + "\n")

    total = 0
    progress = tqdm(test_data)
    for example in progress:
        user_prompt = generate_example(ex_prompt, example)
        if verbose:
            print("=================")
            print(example["question"])
            print(example["options"])
            print("")
        if mode == "student":
            result, executed_ans = pair_student(system_prompt, user_prompt, example["question"], verbose)
        else:
            result, executed_ans = rerun(system_prompt, user_prompt, reruns, verbose)
        
        if verbose:
            print(f"executed answer: {executed_ans}, expected: {example['correct']}")

        tmp = {
            "question": example["question"],
            "correct answer": example["correct"],
            "response": result,
            "executed answer": executed_ans
        }

        if mode == "teacher":
            teacher_feedback = ask_teacher(kwargs["teacher_file"], example["question"], result)
            tmp["teacher feedback"] = teacher_feedback
            if verbose:
                    print("")
                    print(teacher_feedback)
                    print("")
            if not teacher_feedback.lower().startswith("correct"):
                new_result = ask_student(system_prompt, user_prompt, result, teacher_feedback)
                new_ans = safe_execute(new_result, verbose)
                if verbose:
                    print(new_result)
                    print(f"executed answer: {new_ans}, expected: {example['correct']}")
                if new_ans is not None:
                    # if none, prob just said "my old program was correct" or smth
                    executed_ans = new_ans

                tmp["old response"] = tmp["response"]
                tmp["response"] = new_result
                tmp["old executed answer"] = tmp["executed answer"]
                tmp["executed answer"] = new_ans
        elif mode == "mistake":
            if examples_in_ex_prompt < 8:
                teacher_result = ask_mistake(kwargs["mistake_file"], example["question"], result, verbose)
                if teacher_result is not None:
                    while teacher_result.startswith("#") or teacher_result.startswith('\n'):
                        teacher_result = teacher_result[teacher_result.find('\n')+1:]
                    ex_prompt += f"\n\n# Question: {example['question']}\n\n# Solution:\n{teacher_result}"
                    examples_in_ex_prompt += 1
        
        selected_choice = 'A'
        if executed_ans is not None:
            non_null += 1
            selected_choice = select_option(example['question'], example['options'], executed_ans)
            if verbose:
                print(f"selected choice: {selected_choice}")
        
        tmp['selected choice'] = selected_choice

        if selected_choice[:1] == example['correct']:
            correct += 1
            
    
        
        total += 1
        progress.set_postfix({"accuracy": correct/total, "non-null": non_null/total}, refresh=True)
        writer.write(json.dumps(tmp) + '\n')
    
    writer.close()
    print()
    print(f"accuracy: {correct/total}")
    print(f"percent executed: {non_null/total}")

if __name__ == "__main__":
    test_data = load_data(args.data_file)[81: 108]
    prompts = load_prompt(args.system_prompt, args.example_prompt)
    run_test(args.verbose, args.name, test_data, prompts, args.best_of, args.mode, teacher_file=args.teacher_prompt, mistake_file=args.mistake_prompt)

    
        

