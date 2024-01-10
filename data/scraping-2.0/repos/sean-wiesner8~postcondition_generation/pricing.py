from openai import OpenAI

client = OpenAI()

INPUT = "Given the problem description of the programming problem defined below, as well as the definition of a post-condition defined below, create 10 post-conditions in Python to test against an implementation of the programming problem. Before creating these test cases, reiterate what a post-condition is based on the definition described below.\n\nProgramming problem:\n'def remove_kth_element(list1, L):\nWrite a python function to remove the k'th element from a given list.'\n\nDefinition of a post-condition: A post-condition is an assert statement that checks for a condition that should be true regardless of the input.\n\nHere is an example post-condition for an arbitrary programming problem:\n'# Post-condition 1: The output should be a float or an integer.\nassert isinstance(result, (int, float))'"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": INPUT}
    ]
)

message_content = response.choices[0].message.content
prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens
total_tokens = response.usage.total_tokens

print(f"message: {message_content}")
print(f"prompt tokens: {prompt_tokens}")
print(f"completion tokens: {completion_tokens}")
print(f"total tokens: {total_tokens}\n")

input_cost = (prompt_tokens / 1000) * 0.001
output_cost = (completion_tokens / 1000) * 0.002
total_cost = input_cost + output_cost

print(f"Total cost per problem is approx. ${total_cost}")
