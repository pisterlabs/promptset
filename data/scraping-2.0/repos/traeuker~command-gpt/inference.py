import os
from openai import OpenAI
from prompt import get_prompt


with open("/Users/tilman/.openaikey", "r") as file:
    api_key = file.read().strip()

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

def extract_text(output, start_delimiter, end_delimiter):
    start_index = output.find(start_delimiter)
    end_index = output.find(end_delimiter, start_index + len(start_delimiter))

    if start_index != -1 and end_index != -1:
        return output[start_index + len(start_delimiter):end_index].strip()
    return None

def extract_command(output):
    # Define the start and end delimiters for command and notes
    code_start, code_end = '<<', '>>'
    comment_start, comment_end = '++', '--'

    # Use the helper function to extract command and notes
    command = extract_text(output, code_start, code_end)
    notes = extract_text(output, comment_start, comment_end)

    return command, notes


def get_command(instruct, notes, verbose=False):

    initial_prompt = get_prompt()

    messages = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": instruct}
    ]

    if notes:
        messages.append({"role": "user", "content": notes})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    answer = response.choices[0].message.content

    if verbose:
        print(f"\n\nInital prompt: {initial_prompt}\n\n") 
        print(f"Instruction: {instruct}\n\n")
        print(f"Notes: {notes}\n\n")
        print(f"Raw output: {answer}")
    return answer




if __name__ == "__main__":
    ans = get_command('show me all files in this directory', '')
    print(ans)

