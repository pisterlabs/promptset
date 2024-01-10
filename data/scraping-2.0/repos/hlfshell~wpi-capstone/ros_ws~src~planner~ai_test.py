import rclpy
from planner.ai import AI
from planner.llm import OpenAI

from threading import Thread

import time

# from planner.tmp import AI
rclpy.init()

planning_prompt = open("./prompts/planning.prompt", "r").read()
functions_prompt = open("./prompts/functions.prompt", "r").read()
rating_prompt = open("./prompts/rating.prompt", "r").read()

llm = OpenAI(model="gpt-4-1106-preview")
# llm = OpenAI()
# ai = AI(llm)

# print(llm.prompt("Hi there!"))


# prompts = [
#     planning_prompt,
#     functions_prompt,
#     """
# You know about the following rooms and objects. The objects are presented in the format of id# - label

# kitchen:
#     1 - apple
#     22 - coke_can
#     31 - orange
#     12 - wine
#     9 - book
# living_room:
#     14 - tv
#     21 - book
#     19 - coke_can
#     11 - remote_control
# bathroom:
#     13 - toothbrush
#     27 - medicine
#     3 - hairbrush
#     """,
#     """
# Remember, do not reply with anything but python code to accomplish your goal.
# Your objective is to: Get the user a drink
#     """,
# ]
# print(llm.prompt(prompts, temperature=0.7))
# raise "stop"

# response = llm.prompt(prompts, temperature=0.7)
# print("***")
# print(response)
# print("***")
# print(llm.clean_response(response))
# Time the following call
ai = AI(llm, planning_prompt, functions_prompt, rating_prompt)
# import time
executor = rclpy.executors.MultiThreadedExecutor()
executor.add_node(ai)
Thread(target=executor.spin, daemon=True).start()

# start = time.time()
# outputs = ai.generate_plan("Get the user a drink")
# end = time.time()
# print(f"Time taken single: {end - start}")

# print("generating state prompt")
objective = "Get the user a drink"
# print(ai.generate_state_prompt(objective))

start = time.time()
plans = ai.generate_plans(objective)
end = time.time()
print(f"Time taken group: {end - start}")

print("----")
for index, output in enumerate(plans):
    print(f"Plan {index}:")
    print(output)
    print("----")

# # print("Rating:")
# # start = time.time()
# # ratings, reasons = ai.rate_plans(objective, plans)
# # end = time.time()
# # print(f"Time taken rating: {end - start}")
# # print("ratings", ratings)
# # print("reasons", reasons)

print("----")
start = time.time()
plan = ai.get_best_plan(objective, plans)
end = time.time()
print(f"Time taken best plan: {end - start}")
print("===")
print(plan)
