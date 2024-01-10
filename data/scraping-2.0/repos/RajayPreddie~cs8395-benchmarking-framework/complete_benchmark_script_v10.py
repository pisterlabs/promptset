
import os
import json
from openai import Openai
from docx import Document

# OpenAI API setup (ensure you have your API key ready)
openai.api_key = 'YOUR_OPENAI_API_KEY'


# 1. Extraction functions for problems
def extract_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        return data.get('problems', [])

def extract_from_md(md_file):
    with open(md_file, 'r') as file:
        content = file.read()
        problems = content.split("###")[1:]  # Assuming "###" is used to denote a new problem
    return problems

def extract_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        content = file.read()
        problems = content.split("\n\n")  # Assuming problems are separated by a blank line
    return problems

def extract_from_py(py_file):
    with open(py_file, 'r') as file:
        content = file.read()
        problems = content.split('"""')[1::2]  # Assuming problems are described within docstring blocks
    return problems

def extract_from_docx(docx_file):
    doc = Document(docx_file)
    problems = [p.text for p in doc.paragraphs if p.text.strip() != ""]
    return problems

# 2. Extraction functions for test cases
def extract_tests_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        return data.get('tests', [])

def extract_tests_from_py(py_file):
    with open(py_file, 'r') as file:
        content = file.readlines()
        tests = [line for line in content if line.startswith("def test_")]  # Assuming tests start with "def test_"
    return tests

def extract_tests_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        content = file.read().splitlines()
    return content

# 3. OpenAI API request function
def request_solution_from_gpt(problem_description):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=problem_description,
      max_tokens=150
    )
    return response.choices[0].text.strip()

# 4. Testing Function
def test_solution(solution_code, test_code_or_value):
    """Tests the solution against the given test.
    If the test is a code (e.g., a Python file), it runs the solution against the test code.
    If the test is a direct value, it compares the solution's output with the test value.
    """
    try:
        # Assuming the test code expects a function called 'solution' 
        # that it will call and check for correctness.
        local_vars = {}
        exec(solution_code + "\n" + test_code_or_value, {}, local_vars)
        if 'test_result' in local_vars:
            return local_vars['test_result']
        else:
            return False
    except Exception as e:
        print(f"Testing error: {{e}}")
        return False

# 5. Scoring Logic
def compute_score(problems, results):
    total_weight = sum([problem.get('weight', 1) for problem in problems])
    total_score = sum([result.get('test_result', 0) * problems[idx].get('weight', 1) for idx, result in enumerate(results)])
    scores = {
        "overall": total_score / total_weight,
        "by_category": {}  # Extend this for category-based scoring
    }
    return scores

# 6. Main orchestration function
def main_benchmark(folder_path, test_folder_path):
    # Extract problems and tests
    problems = extract_from_folder(folder_path)
    tests = extract_tests_from_folder(test_folder_path)
    
    results = []
    for idx, problem in enumerate(problems):
        # Request solution from ChatGPT
        solution = request_solution_from_gpt(problem['problem_description'])
        
        # Test the solution
        test_result = test_solution(solution, tests[idx])
        
        # Append results
        results.append({
            "problem_id": problem.get("problem_id"),
            "name": problem.get("name"),
            "category": problem.get("category"),
            "solution": solution,
            "test_result": test_result
        })
    
    # Compute scores
    scores = compute_score(problems, results)
    
    # Save results to JSON
    output = {
        "results": results,
        "scores": scores
    }
    with open("results.json", "w") as json_file:
        json.dump(output, json_file, indent=4)

# For execution
if __name__ == "__main__":
    folder_path = input("Enter the path to the problems folder: ")
    test_folder_path = input("Enter the path to the tests folder: ")
    main_benchmark(folder_path, test_folder_path)


# Extraction functions for entire folders
def extract_from_folder(folder_path):
    problems = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".json"):
            problems.extend(extract_from_json(file_path))
        elif filename.endswith(".md"):
            problems.extend(extract_from_md(file_path))
        elif filename.endswith(".txt"):
            problems.extend(extract_from_txt(file_path))
        elif filename.endswith(".py"):
            problems.extend(extract_from_py(file_path))
        elif filename.endswith(".docx"):
            problems.extend(extract_from_docx(file_path))
    return problems

def extract_tests_from_folder(test_folder_path):
    tests = []
    for filename in os.listdir(test_folder_path):
        file_path = os.path.join(test_folder_path, filename)
        if filename.endswith(".json"):
            tests.extend(extract_tests_from_json(file_path))
        elif filename.endswith(".py"):
            tests.extend(extract_tests_from_py(file_path))
        elif filename.endswith(".txt"):
            tests.extend(extract_tests_from_txt(file_path))
    return tests



def generate_test_or_check_solution(problem_description, solution_code):
    """Generates a test or checks the solution using ChatGPT.
    First, it tries to generate a test for the given problem.
    If unable to generate a meaningful test, it checks if the solution is correct for the given problem.
    """
    # Request ChatGPT to generate a test for the problem
    test_prompt = f"Generate a test for this problem: {{problem_description}}"
    test_code = request_solution_from_gpt(test_prompt)
    
    # If the generated test is not meaningful or too short, validate the solution directly
    if len(test_code) < 15:  # Threshold can be adjusted
        validation_prompt = f"Is this solution correct for the problem: {{problem_description}}?\n{{solution_code}}"
        validation_response = request_solution_from_gpt(validation_prompt)
        return "True" in validation_response  # Assumes ChatGPT returns "True" or "False" in its response
    else:
        return test_solution(solution_code, test_code)

