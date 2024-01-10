import openai
import os
import re
import json

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "Your OpenAI API key" 
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_chatgpt_response(prompt, model='gpt-3.5-turbo', temperature=0):
    """
    Returns the response from ChatGPT for a given prompt
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt,
        temperature=temperature,
    )
    return response


def get_code_from_response(text, language):
    if "```" not in text:
        return text.strip()

    if f"```{language}" in text:
        pattern = rf'```{language}(.*?)```'
    else:
        pattern = r'```(.*?)```'

    code_blocks = re.findall(pattern, text, re.DOTALL)
    return ''.join(code_blocks).strip()


def generate_code_by_feedback_with_static_analysis(output_dir, lang="python"):
    """
    Generates code from ChatGPT for a given code task
    """
    tasks_meta_path = r"path/to/leetcode_tasks/leetcode.json"
    issued_code_meta_path = r"path/to/data/chatgpt_generated_code/{}.json".format(lang)

    with open(tasks_meta_path, 'r') as f:
        tasks_meta_lists = json.load(f)

    with open(issued_code_meta_path, 'r') as f:
        issued_code_meta_lists = json.load(f)

    for task_meta in tasks_meta_lists:
        task_id = task_meta['id']
        task_name = task_meta['name']
        task_description = task_meta['task_description']

        issued_code_meta = None
        for issued_code_meta in issued_code_meta_lists:
            if issued_code_meta['id'] == task_id:
                break

        error = issued_code_meta['error'] if issued_code_meta is not None else None
        quality = issued_code_meta['is_quality_issue'] if issued_code_meta is not None else None

        if (error is not None and error != "") or (quality is not None and quality == 1):
            messages_prompt = f"Please provide a code implementation of the following description:\n{task_description}"
            template_key = f"{lang}_template"
            if template_key in task_meta:
                messages_prompt += f"\nProvide a valid {lang} code with this template:\n{task_meta[template_key]}"

            messages = [{"role": "system", "content": f"Your task is to write a {lang} program"}]
            messages.append({"role": "user", "content": messages_prompt})
            messages.append({"role": "assistant", "content": f"Here is code:\n{issued_code_meta['generated_code']}"})

            prompts_with_static_info = "The generated code contains the following quality issues:\n" + issued_code_meta['error_info'] + issued_code_meta["quality_info"]
            prompts_with_static_info += "\nPlease provide a better code implementation as expected by the task description."
            messages.append({"role": "user", "content": prompts_with_static_info})

            code = get_code_from_response(response.choices[0].message.content, lang)
            if lang == "python":
                output_file = os.path.join(output_dir, f"{task_id}-{task_name}.py")
            else:
                output_file = os.path.join(output_dir, f"{task_id}-{task_name}.java")

            with open(output_file, 'w') as f:
                f.write(code)

if __name__ == "__main__":
    output_directory = "path/to/data/results/"
    generate_code_by_feedback_with_static_analysis(output_directory, lang="python")