import filecmp
import os
import openai
import shutil
import inquirer
import subprocess

from git import Repo

openai.api_key = os.getenv("OPENAI_API_KEY")
gpt_model = "gpt-3.5-turbo-16k"
ignore_file_suffix_list = ['spec.ts']
# Create a repo object
repo_path = "<ROOT OF YOUR REPOS PATH>"

def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# get a list of subdirectories under repo_path
subdirectories = get_subdirectories(repo_path)

# ask the user to select a subdirectory
dir_questions = [
    inquirer.List('subdir',
                  message="Select a subdirectory",
                  choices=subdirectories,
                 ),
]
dir_answers = inquirer.prompt(dir_questions)
selected_subdir = dir_answers['subdir']

# use the selected subdirectory for the rest of your code
repo_path = os.path.join(repo_path, selected_subdir)

def compare_files_w_beyondcompare(file1, file2):
    try:
        result = subprocess.run(['bcompare', file1, file2], check=True)
        print('Comparison completed successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Error during comparison: {e}')

def ignore_folders(dir, files):
    ignored = ['node_modules', '.git', 'public']
    return [name for name in files if name in ignored]

def compare_files(file1, file2):
    if not filecmp.cmp(file1, file2, shallow=False):
        return (file1, file2)
    else:
        return None

def compare_directories(dir1, dir2):
    differences = []

    dcmp = filecmp.dircmp(dir1, dir2)

    for name in dcmp.common_files:
        if any(name.endswith(ignore) for ignore in ignore_file_suffix_list):
            continue

        file1 = os.path.join(dir1, name)
        file2 = os.path.join(dir2, name)
        diff = compare_files(file1, file2)
        if diff is not None:
            differences.append(diff)

    for sub_dcmp in dcmp.subdirs.values():
        sub_dir_diff = compare_directories(
            sub_dcmp.left,
            sub_dcmp.right
        )
        differences.extend(sub_dir_diff)

    return differences

def compare_branch_files(branch1, branch2):
    print("branch1",branch1)
    print("branch2",branch2)

    differences = []
    repo_name = repo_path.split("\\")[-1]

    temp_dir1 = f"{repo_name}_temp_{branch1}"
    checkout_dir1 = os.path.join(temp_dir1, "checkout")

    temp_dir2 = f"{repo_name}_temp_{branch2}"
    checkout_dir2 = os.path.join(temp_dir2, "checkout")

    # Delete existing directories if they exist
    for dir_path in [checkout_dir1, checkout_dir2]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    os.makedirs(checkout_dir1, exist_ok=True)
    os.makedirs(checkout_dir2, exist_ok=True)

    repo = Repo(repo_path)
    original_branch = str(repo.active_branch)

    try:
        repo.git.checkout(branch1)
        shutil.copytree(repo_path, checkout_dir1, ignore=ignore_folders, dirs_exist_ok=True)

        repo.git.checkout(branch2)
        shutil.copytree(repo_path, checkout_dir2, ignore=ignore_folders, dirs_exist_ok=True)

        if os.path.exists(checkout_dir1) and os.path.exists(checkout_dir2):
            differences = compare_directories(checkout_dir1, checkout_dir2)
    finally:
        repo.git.checkout(original_branch)


    return differences, checkout_dir1, checkout_dir2


repo = Repo(repo_path)
branches = [str(branch) for branch in repo.branches]
questions = [
    inquirer.List('branch1',
                  message="Select the first branch",
                  choices=branches,
                 ),
    inquirer.List('branch2',
                  message="Select the second branch",
                  choices=branches,
                 ),
]

answers = inquirer.prompt(questions)
branch1 = answers['branch1']
branch2 = answers['branch2']

differences, checkout_dir1, checkout_dir2 = compare_branch_files(branch1, branch2)


def save_files_to_disk(file1, file2, content1, content2):
    with open(f"{branch1}_{os.path.basename(file1)}_debug.txt", 'w') as f1_debug:
        f1_debug.write(content1)
    with open(f"{branch2}_{os.path.basename(file2)}_debug.txt", 'w') as f2_debug:
        f2_debug.write(content2)

def save_ai_response(file1, file2, response):
    with open(f"ai_{os.path.basename(file1)}_vs_{os.path.basename(file2)}_response.txt", 'w') as f_response:
        f_response.write(response)

def normalize_spaces(text):
    return ' '.join(text.splitlines())

def create_prompt(file1, file2, base1, base2):
    relative_path1 = os.path.relpath(file1, base1)
    relative_path2 = os.path.relpath(file2, base2)

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        content1 = f1.read()
        content2 = f2.read()

    save_files_to_disk(file1, file2, content1, content2)

    normalized_content1 = normalize_spaces(content1)
    normalized_content2 = normalize_spaces(content2)

    prompt = f"There are two pieces of code from {branch1} and {branch2} environments. Please analyze both code files and briefly explain what the developer tried to change from {branch1} to {branch2} environment, ignore not changes. The first code is from {relative_path1} in {branch1} and the second code is from {relative_path2} in {branch2}.\n\n{branch1} ENV:\n{normalized_content1}\n\n{branch2} ENV:\n{normalized_content2}"
    return prompt

def print_diff(file1, file2, base1, base2):
    prompt = create_prompt(file1, file2, base1, base2)
    summary = ask_gpt(prompt)
    save_ai_response(file1, file2, summary)
    print(summary)

def ask_gpt(prompt):
    response = openai.ChatCompletion.create(
        model= gpt_model,
        messages=[
            {"role": "system", "content": "I am a full-stack code reviewer and code analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature = 1,
        max_tokens=10000
    )
    print("Total Used Token:",response['usage'].total_tokens)
    
    choices = response['choices']
    if choices and len(choices) > 0:
        return choices[0]['message']['content']
    else:
        return "No response generated."

if differences:
    # Ask the user to choose the scope of comparison first
    compare_scope_questions = [
        inquirer.List('scope',
                      message="Do you want to compare individual files or whole folders?",
                      choices=['Files', 'Folders'],
                     ),
    ]
    compare_scope_answers = inquirer.prompt(compare_scope_questions)
    selected_scope = compare_scope_answers['scope']

    # Then present the appropriate comparison tool options
    if selected_scope == 'Folders':
        compare_files_w_beyondcompare(checkout_dir1, checkout_dir2)
    else:  # Files
        # Ask the user to choose a tool for comparison
        tool_questions = [
            inquirer.List('tool',
                          message="Which tool would you like to use for comparison?",
                          choices=['Beyond Compare', 'ChatGPT'],
                         ),
        ]
        tool_answers = inquirer.prompt(tool_questions)
        selected_tool = tool_answers['tool']

        if selected_tool == 'Beyond Compare':
            for file1, file2 in differences:
                questions = [inquirer.List('file',
                                            message="Which file do you want to see the difference?",
                                            choices=[f"{branch1}: {os.path.basename(file1)} vs {branch2}: {os.path.basename(file2)}" for file1, file2 in differences],
                                        ),
                            ]
                answers = inquirer.prompt(questions)
                selected = answers['file']
                for file1, file2 in differences:
                    if f"{branch1}: {os.path.basename(file1)} vs {branch2}: {os.path.basename(file2)}" == selected:
                        compare_files_w_beyondcompare(file1, file2)
                        break
        else:  # ChatGPT
            for file1, file2 in differences:
                questions = [inquirer.List('file',
                                            message="Which file do you want to see the difference?",
                                            choices=[f"{branch1}: {os.path.basename(file1)} vs {branch2}: {os.path.basename(file2)}" for file1, file2 in differences],
                                        ),
                            ]
                answers = inquirer.prompt(questions)
                selected = answers['file']
                for file1, file2 in differences:
                    if f"{branch1}: {os.path.basename(file1)} vs {branch2}: {os.path.basename(file2)}" == selected:
                        print_diff(file1, file2, checkout_dir1, checkout_dir2)
                        break
else:
    print("No differences found.")
