import os
import openai
import json
from typing import Union

openai.api_key = ""

def analyze_and_optimize_code(code: str, file_type: str) -> str:
    # prompt = f"Analyze and optimize the following {file_type} code for better performance and readability:\n\n{code}\n\nOptimized Code:"
    prompt = f"Analyze and optimize the following {file_type} code for better performance and readability:\n\n{code}\n\nIf there are any optimizations or changes, provide the optimized code and an explanation of why the changes were made. If no changes are necessary, explain why the original code is already well-written and optimized.\n\n"
    
    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt, max_tokens=1024, n=1, stop=None, temperature=0.5
    )
    optimized_code = response.choices[0].text.strip()

    return optimized_code


def process_file(file_path: str):
    file_type = ""
    if file_path.endswith(".ts") or file_path.endswith(".tsx"):
        file_type = "TypeScript"
    elif file_path.endswith(".js"):
        file_type = "JavaScript"
    elif file_path.endswith(".json"):
        file_type = "JSON"

    if not file_type:
        return

    with open(file_path, "r") as file:
        content = file.read()
        


    optimized_content_and_explanation = analyze_and_optimize_code(content, file_type)
    # with open(file_path, "w") as file:
      #  file.write(optimized_content)
    # Extract the explanation text from the optimized_content_and_explanation
    explanation_start = optimized_content_and_explanation.find("Explanation:")
    explanation = optimized_content_and_explanation[explanation_start:]
    optimized_content = optimized_content_and_explanation[:explanation_start].strip()

    # Print the original and optimized content in markdown format with the explanation as header text
    print(f"Optimization suggestions for {file_path}:\n")
    print(f"Original {file_type} code:\n")
    print(f"```{file_type}\n{content}\n```\n")
    print(f"Optimized {file_type} code:\n")
    print(f"```{file_type}\n{optimized_content}\n```\n")
    print(f"#### {explanation}\n")


def analyze_project_files(root_dir: str):
    for folder, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(folder, file)
            process_file(file_path)

if __name__ == "__main__":
    project_directory = "/Users/victorchavarro/Documents/dev/personal/test/"
    analyze_project_files(project_directory)