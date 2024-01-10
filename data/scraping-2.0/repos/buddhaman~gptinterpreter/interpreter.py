import openai
import json

openai.api_key = "..."

output = ""

def do_instruction(state, instruction):
    prompt = """
    You manipulate internal state, the state will be in json form.

    The current state is: 
    """

    prompt += state
    prompt += """

    The instruction to manipulate state is:

    """
    prompt += instruction 
    prompt += """
    The following is some example output, always include all keys:
    {
        "mem0": ["<example content>"],
        "mem1": "Example json values",
        "mem2": "Example",
        "mem3": "Example",
        "output": "Example",
    }

    Give ONLY this Json object as output, no additional text. Leave "output" empty unless instructed otherwise. Output is always a string. 
    """

    print(prompt)

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a computer. You output new state based on input."},
            {"role": "user", "content": prompt},
        ]
    )

    content = completion["choices"][0]["message"]["content"]
    return content

state = """
{
    "mem0": "",
    "mem1": "",
    "mem2": "",
    "mem3": "",
}
"""

instructions = [
    """Fill every memory slot with a cool date idea. Output all these ideas in a structured way.""",
    """Replace mem3 with a horrible joke, replace mem1 with the average idea of mem0 and mem1, output your favorite horse breed.""",
    """In "output", combine all memory into something awful."""
]

output = ""

for instruction in instructions:
    state_str = do_instruction(state, instruction)
    print("RAW OUTPUT  = ", state_str, "\n------\n")
    state_dict = json.loads(state_str)
    state = json.dumps({
        "mem0": state_dict["mem0"],
        "mem1": state_dict["mem1"],
        "mem2": state_dict["mem2"],
        "mem3": state_dict["mem3"],
    })
    output += state_dict["output"]
    output += "\n"
    print(state)
    print("Output: ", output)

with open("output.txt", "w") as file:
    file.write(output)
