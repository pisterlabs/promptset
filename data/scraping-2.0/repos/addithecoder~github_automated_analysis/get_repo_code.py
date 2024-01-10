#!/usr/bin/env python
# coding: utf-8

# # Declaration
# I have prepared the project below keeping in mind the feedback points that Mercor had provided me for my previous project.
# 1) I have prepared different functions which are seperated and each function is designed to perform distinct function.
# 2) Added comments to document the purpose and functionality of each function.
# 3) Other points such as, preprocessing and adding error handelling which were adviced to be added have all been added in this project.
# 
# I hope I worked to your satisfaction. 
# Looking forward to your response.
# Thank You

# Iimport all the requires packages and Declaring the variables

# In[8]:


import requests
import re
import openai
import time
MAX_FILE_SIZE = 10 * 1024 * 1024
API_CALL_DELAY = 2  # Delay in seconds between API calls


# Function 'get_user_repositories' is to get the repositories from the GitHub url given by the user

# In[9]:


def get_user_repositories(user_url):
    # Extract username from the GitHub user URL
    username = user_url.split('/')[-1]

    # API endpoint for fetching user's repositories
    api_url = f"https://api.github.com/users/{username}/repos"

    try:
        # Send GET request to GitHub API
        response = requests.get(api_url)

        # Check if the request was successful
        if response.status_code == 200:
            repositories = response.json()

            # Extract relevant repository information
            extracted_repositories = []
            for repo in repositories:
                extracted_repo = {
                    'name': repo['name'],
                    'owner': repo['owner']['login'],
                    # Add more relevant repository metadata if needed
                }
                extracted_repositories.append(extracted_repo)

            return extracted_repositories
        else:
            # Handle unsuccessful API response
            print(f"Failed to fetch repositories. Status Code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        # Handle request exceptions
        print("Error occurred during API request:", e)
        return None


# Extract all the codes from the repositories

# In[10]:


def extract_repository_code(repo_owner, repo_name):
    # API endpoint for fetching repository contents
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"

    try:
        # Send GET request to GitHub API
        response = requests.get(api_url)

        # Check if the request was successful
        if response.status_code == 200:
            contents = response.json()

            # Iterate over the contents to find files
            for item in contents:
                if item['type'] == 'file':
                    file_url = item['download_url']
                    file_content = requests.get(file_url).text

                    # Do something with the file content (e.g., print or save it)
                    #print(f"File: {item['path']}")
                    #print(f"Content:\n{file_content}\n")

        else:
            # Handle unsuccessful API response
            print(f"Failed to fetch repository contents. Status Code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        # Handle request exceptions
        print("Error occurred during API request:", e)
        
    repositories_complexity = {}  # Dictionary to store repository complexity scores
    # Iterate over the contents to find Python files
    for item in contents:
        if item['type'] == 'file' and item['name'].endswith('.py'):
            file_url = item['download_url']
            file_size = int(item['size'])

            if file_size <= MAX_FILE_SIZE:
                file_content = requests.get(file_url).text

                # Preprocess the file content to keep only Python code
                python_code = extract_python_code(file_content)

                if python_code:
                    # Generate prompt for complexity evaluation
                    prompt = generate_prompt(python_code)

                    # Evaluate complexity using GPT with delay
                    complexity = evaluate_complexity_with_delay(prompt)

                    # Store the complexity with repository name
                    repositories_complexity[repo_name] = complexity
                    print(complexity)
            else:
                print(f"Skipping file {item['path']} due to large size")

    # Check if there are repositories with Python code
    if repositories_complexity:
        # Identify the repository with the highest complexity score
        most_complex_repository = max(repositories_complexity, key=repositories_complexity.get)
        justification = generate_justification(repositories_complexity)
        return most_complex_repository, justification
    else:
        return None, "No repositories with Python code found"


# Function to extract python code from files in repository

# In[11]:


def extract_python_code(file_content):
    # Remove comments and blank lines
    code_lines = [line.strip() for line in file_content.split('\n') if line.strip() and not line.strip().startswith('#')]

    # Join the code lines
    python_code = '\n'.join(code_lines)

    # Filter out non-Python code (e.g., docstrings, imports, etc.)
    python_code = filter_python_code(python_code)

    return python_code


# Function to generate prompt for complexity evaluation

# In[12]:


def generate_prompt(python_code):
    # Define your prompt generation strategy here
    prompt = f"Given the following Python code, determine its technical complexity:\n\n{python_code}"
    return prompt


# This function is to evaluate the compexity using GPT 

# In[13]:


def evaluate_complexity_with_delay(prompt):
    # Set up OpenAI API credentials
    openai.api_key = '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'

    # Define the parameters for the GPT call
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,  # Adjust this value based on your requirements
        temperature=0.5,  # Adjust this value based on your requirements
        n=1,
        stop=None,
        #temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract the generated completion from the API response
    completion = response.choices[0].text.strip()

    # Parse the completion and extract the complexity score
    complexity = parse_complexity(completion)

    # Delay between API calls to avoid rate limiting
    time.sleep(API_CALL_DELAY)

    return complexity


# This function is to parse the complexity score from the GPT completion

# In[14]:


def parse_complexity(completion):
    # Extract the complexity score from the completion text
    match = re.search(r"Complexity Score: ([\d.]+)", completion)
    if match:
        complexity_score = float(match.group(1))
        return complexity_score

    # Return a default value if the complexity score cannot be parsed
    return 0.0


# Generate the justification after comparing the complexity of each repository

# In[15]:


def generate_justification(repositories_complexity):
    # Sort repositories by complexity in descending order
    sorted_repositories = sorted(repositories_complexity.items(), key=lambda x: x[1], reverse=True)

    # Get the most complex repository
    most_complex_repository = sorted_repositories[0]

    # Generate a justification based on the complexity score
    justification = "The repository with the most technical complexity is:\n"
    justification += f"Repository Name: {most_complex_repository[0]}\n"
    justification += f"Complexity Score: {most_complex_repository[1]}\n"

    return justification


# This function is used to filter out the non python code.

# In[16]:


def filter_python_code(python_code):
    # Define patterns to filter out non-Python code
    patterns = [
        r'^from .* import .*$',  # Imports
        r'^import .*$',          # Imports
        r'^def .*\):$',          # Function definitions
        r'^class .*:$',          # Class definitions
        r'^@.*$',                # Decorators
        r'^.*#.*$',              # Comments
    ]

    # Apply the patterns and filter the code
    for pattern in patterns:
        python_code = re.sub(pattern, '', python_code, flags=re.MULTILINE)

    # Remove leading and trailing whitespaces
    python_code = python_code.strip()

    return python_code


# This function is responsible to create the url for the most complex repository after comparing the complexities.

# In[17]:


def get_repository_link(repo_owner, repo_name):
    return f"https://github.com/{repo_owner}/{repo_name}"


# # Main function

# In[18]:


def main():
    # Get user GitHub URL from the user
    user_url = input("Enter GitHub user URL: ")

    # Retrieve user's repositories
    repositories = get_user_repositories(user_url)

    # Extract code from each repository
    if repositories:
        for repo in repositories:
            print(f"Extracting code from repository: {repo['name']}")
            extract_repository_code(repo['owner'], repo['name'])
            
	#Get the most complex repository by calculating the complexity using GPT and print the justification 
    #and the link to get to the repository.
    most_complex_repository, justification = get_user_repositories(user_url)
    if most_complex_repository:
        repo_owner = most_complex_repository['owner']
        repo_name = most_complex_repository['name']
        repo_link = get_repository_link(repo_owner, repo_name)
        print(f"Most complex repository: {repo_owner}/{repo_name}")
        print(f"Justification: {justification}")
        print(f"Repository Link: {repo_link}")
    else:
        print(justification)


# Entry point
if __name__ == '__main__':
    main()


# In[ ]:




