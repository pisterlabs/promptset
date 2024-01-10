import re
from openai import OpenAI

client = OpenAI()
from tqdm import tqdm
from selfmodifai.agents.fine_tunable_agents.replace_code import replace_code
from selfmodifai.agents.fine_tunable_agents.codellama_generate import codellama_generate


def open_source_agent():
    with open("prompts/nanogpt/researcher.txt", "r") as f:
        prompt = f.read()

    for i in tqdm(range(20)):
        output_str = codellama_generate(prompt)
        if len(set(output_str)) > 1:
            break

    # training_file_path = "/selfmodifai/selfmodifai-gpt-dev/gpt_dev/train.py"
    training_file_path = "gpt-dev/train.py"

    with open(training_file_path, "r") as f:
        training_file = f.read()

    engineer_prompt = (
        "This is my training script:\n"
        + training_file
        + "\n\nCan you add these changes to it? Only give me the functions that are changed:\n"
        + output_str
    )
    gpt_api_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": engineer_prompt},
    ]
    engineer_response = client.chat.completions.create(model="gpt-4", messages=gpt_api_messages)
    engineer_response_content = engineer_response["choices"][0]["message"]["content"]

    print("GPT-4:\n", engineer_response_content)

    pattern = r"```python\n(.*?)\n```"
    er_search = re.search(pattern, engineer_response_content, re.DOTALL)
    if er_search:
        er_code = er_search.group(1)
        replace_code(er_code)
