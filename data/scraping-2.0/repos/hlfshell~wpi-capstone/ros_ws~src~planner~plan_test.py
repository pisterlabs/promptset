import rclpy
from planner.ai import AI
from planner.llm import OpenAI, PaLM

import ast

from concurrent.futures import ThreadPoolExecutor, wait

import time

count = 100

rclpy.init()

planning_prompt = open("./prompts/planning.prompt", "r").read()
functions_prompt = open("./prompts/functions.prompt", "r").read()
objective = "Get the user a drink"
instructions_prompt = (
    "Remember, do not reply with anything but python code to accomplish your goal."
)
objective_str = f"Your objective is to: {objective}"


models = {
    "gpt4": OpenAI(model="gpt-4-1106-preview"),
    "gpt3.5-turbo": OpenAI(),
    "PaLM": PaLM(),
}

chosen_model = "PaLM"
print(f"Using model: {chosen_model}")
llm = models[chosen_model]

ai = AI(llm)

state_prompt = ai.generate_state_prompt(objective)

prompts = [
    planning_prompt,
    functions_prompt,
    state_prompt,
    instructions_prompt,
    objective_str,
]

times = []
plans = []


def generate():
    global times
    global plans
    start = time.time()
    response = llm.prompt(prompts, temperature=0.7)
    end = time.time()
    plans.append(llm.clean_response(response))
    times.append(end - start)


def measure_performance():
    futures = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(count):
            future = executor.submit(generate)
            futures.append(future)
    print("Waiting on futures", len(futures))
    wait(futures)


measure_performance()

compile_fail = 0
actual_code = 0
lengths = []
lines = []

for plan in plans:
    lengths.append(len(plan))
    lines.append(plan.split("\n"))
    try:
        ast.parse(plan)
    except Exception:
        compile_fail += 1
        continue

    # Check to see if we have at least three hits
    # of our functions
    functions = [
        "move_to_object",
        "move_to_human",
        "move_to_room",
        "pickup_object",
        "give_object",
        "do_i_see",
        "look_around_for",
    ]
    hits = 0
    for function in functions:
        if function in plan:
            hits += 1
            if hits >= 3:
                actual_code += 1
                break

print(f"Completed {len(plans)} / {count}")
count = len(plans)

print(f"Compile fail: {compile_fail}/{count}= {compile_fail/count:.2f}")
print(f"Actual code: {actual_code}/{count}= {actual_code/count:.2f}")
print(f"Average length: {sum(lengths)/len(lengths)}")
print(f"Longest length: {max(lengths)}")
print(f"Shortest length: {min(lengths)}")
print(f"Average lines: {sum(len(line) for line in lines)/len(lines)}")
print(f"Longest lines: {max(len(line) for line in lines)}")
print(f"Shortest lines: {min(len(line) for line in lines)}")
print(f"Average time: {sum(times)/len(times)}")
print(f"Longest time: {max(times)}")
print(f"Shortest time: {min(times)}")
