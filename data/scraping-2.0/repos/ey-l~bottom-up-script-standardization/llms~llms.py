import openai
import os
import sys
import random
import re
sys.path.append(f'{sys.path[0]}/..')
openai.api_key = "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"


#prompt = "What is science?"
model = "gpt-4"
N = 3 # beam size
# Define a regular expression pattern to match the content
pattern = r"```python(.*?)```"

def load_scripts(root):
    # Get all file paths
    verbose = True
    scripts = []
    #print(sys.path)
    for path, subdirs, files in os.walk(root):
        for name in files:
            # Load scripts
            if name.split('.')[-1] == 'py' and name.split('.')[0] not in ['0', '1', '2']:
                with open(os.path.join(path, name), encoding='utf-8') as f:
                    code = f.read()
                    code = code.replace("_data/input/", "data/input/")
                    code = code.replace("data/input/", f"{sys.path[-1]}/data/input/")
                scripts.append(code)
    if verbose:
        print("Number of scripts:", len(scripts))
    return scripts

# Function that calls the GPT-4 API
def ask_GPT_model(model, system_intel, prompt): 
    try:
        result = openai.ChatCompletion.create(model = model,
                                    messages=[{"role": "system", "content": system_intel},
                                            {"role": "user", "content": prompt}],
                                            n=N,
                                            temperature=0)
        return [i['message']['content'] for i in result['choices']]
    except Exception as e:
        print(e)
        return []
    # display(Markdown(result['choices'][0]['message']['content']))

datasets = ["house-prices-advanced-regression-techniques", "titanic",
            "competitive-data-science-predict-future-sales", "nlp-getting-started", 
            "spaceship-titanic", "uciml_pima-indians-diabetes-database", ]
datasets = ["titanic"]
#model = "gpt-3.5-turbo"
model = 'gpt-4'
system_intel = f"You are {model}, answer my questions as if you were an expert in data preprocessing." # change it as needed
for dataset in datasets:
    script_directory = f"/Users/eylai/Projects/lucid-script/scratch/gpt-4/{dataset}"
    dirctory_save_to = f"/Users/eylai/Projects/lucid-script/scratch/gpt-4-promp-1/{dataset}"
    scripts = load_scripts(script_directory)
    scripts_sampled = random.sample(scripts, 5)

    # Get all folders in the script directory
    folders = [f for f in os.listdir(script_directory) if os.path.isdir(os.path.join(script_directory, f))]
    for folder in folders:
        # Get all files in the folder
        files = [f for f in os.listdir(os.path.join(script_directory, folder)) if os.path.isfile(os.path.join(script_directory, folder, f))]
        for file in files:
            # Make the directory
            if not os.path.exists(os.path.join(dirctory_save_to, folder)):
                os.makedirs(os.path.join(dirctory_save_to, folder))
            # Read the file
            with open(os.path.join(script_directory, folder, file), encoding='utf-8') as f:
                code = f.read()
                code = code.replace("_data/input/", "data/input/")
                code = code.replace("data/input/", f"{sys.path[-1]}/data/input/")
            with open(os.path.join(dirctory_save_to, folder, file), 'w') as f:
                f.write(code)
            # Call the function above
            prompt = f"I want to preprocess a dataset from a Kaggle competition, and wrote the following Python script for that:\n```python\n{code}\n```\nHowever, I'm unsure if I'm missing any steps or should leave some steps out. Specifically, I want my Python script above to be similar to what other people are doing. I gave you three scripts of other people below. Please alter my Python program above, and include or exclude steps based on what the other scripts below do. Your output should consist of one python script that compiles and runs, nothing more.:\n```python\n{scripts_sampled[0]}\n```\n```python\n{scripts_sampled[1]}\n```\n\n```python\n{scripts_sampled[2]}\n```\nNow output a Python script. You should only return straight-line Python code."
            prompt = f"My goal is to train a machine learning model for the Kaggle competition, Titanic. The model aims to predict the chances of survival of a passenger on the Titanic using other traits. In the dataset, each training entry is a row, consisting of the following columns: passenger Id (int), survived (target prediction column, boolean), pclass(enum 1, 2, 3), name (string), sex (enum (male, female)), age (float), sibsp (boolean), parch (boolean), ticket (string), fare (float), cabin (string), embarked (enum(C, S)). Using the information above, please check this data pre-processing script.\n```python\n{code}\n```\nIn particular, what are the steps that looks abnormal compared to other people's scripts below? What are some steps that are generally included in other people's scripts that are not in mine?\n```python\n{scripts_sampled[0]}\n```\n```python\n{scripts_sampled[1]}\n```\n\n```python\n{scripts_sampled[2]}\n```\n```python\n{scripts_sampled[3]}\n```\n```python\n{scripts_sampled[4]}\n```\n\nNow output a Python script. You should only return straight-line Python code."
            prompt = f"I'm working on kaggle Titanic competition: https://www.kaggle.com/competitions/titanic/code. I'm not sure about whether my script is correct, i.e., my script may miss some crucial steps that are present in the collection, or my script has anomaly steps that are not used by other competitors. Can you check the following script:\n```python\n{code}\n```\nCan you directly optimize my script? \nNow output a Python script. You should only return straight-line Python code."
            prompt = f"Imagine you are trying to write a python script for a data analysis job. You are given a csv file that include passenger information of the Titanic like name, age, gender, socio-economic class, etc., and whether they survived or not. You want to use this csv to train a machine learning model to predict whether someone is likely to survive or not. Write a python script that does this. Here is a starting point: \n```python{code}```\n"
            prompt = f"Given a collection of Python scripts from the Kaggle Titanic competition \n```python\n{scripts_sampled[0]}\n```\n```python\n{scripts_sampled[1]}\n```\n\n```python\n{scripts_sampled[2]}\n```\n```python\n{scripts_sampled[3]}\n```\n```python\n{scripts_sampled[4]}\n```\n, I am looking to improve my current script. Here is my current script:\n```python\n{code}\n```\nPlease analyze the collection of scripts and suggest improvements or additional steps that could potentially enhance the performance of my script. The improvements could be in terms of model accuracy or any other aspect that you find relevant. Please provide the improved Python script.\nNow output a Python script. You should only return straight-line Python code."
            prompt = f"Read the following collection of python scripts (https://www.kaggle.com/competitions/titanic/code.) from a Kaggle contest on Titanic dataset and extract common patterns and best practices in them. Input:\n```python\n{code}\n```\n\nNow output a Python script. You should only return straight-line Python code."
            prompt = f"Now I will give you another script for the same contest. \n Your job is to improve this script in terms of <runtime performance|accuracy|recall> using the common patterns and best practices you have learned from the collection.\n You are allowed to add and delete steps.  \n Step is defined as any valid python statement.   \n Make sure your output is a syntactically correct python script.\n Here is the input: \n```python\n{code}\n```\n\nNow output a Python script. You should only return straight-line Python code."
            prompt = f"—\n I am working on a Kaggle competition using the titanic dataset. I want you help me with the preprocessing of the dataset, using standard winning methods.\n For example, automatically add things like:\n - Imputation\n - Normalization.\n - Anything else that might help.\n Also remove bad, useless, or \"illegal\" steps.\n \"Illegal\" steps are things like including the target (survived or not) in the final output. Likewise faking a column strongly correlated (like survived + 1). I want you to remove these bad steps, and other pointless steps.\n —\nHere are a few tips:\n- Assume there is a local file called \"titanic.csv\" containing the training data.\n- Use pandas to read it.\n- Use anything you see fit for preprocessing.\n—\nI will give you as input a partially written script (potentially empty). Your output should be a complete preprocessing script, including all the imports (e.g., using pandas). Format your output as follows:(START)\n```python```\n(END)\n—\nHere is the input:\n```python\n{code}\n```\n\nNow output a Python script. You should only return straight-line Python code."
            prompt = f"Pretend you are an expert data analyst with extensive knowledge and expertise in data analysis, data cleaning, data wrangling, and machine learning. Write a Python program to achieve high accuracy on the Titanic - Machine Learning from Disaster challenge on Kaggle. You have analyzed the wealth of information that is available online for solving this challenge, such as code repositories, blog articles, tutorials, etc. List a few advanced machine learning models that others have used for this challenge. What is the reported accuracy of these methods? Use the model with the highest accuracy for the code. Similarly, what data cleaning and wrangling methods are used for this challenge? Use the techniques that are widely used by others to achieve high accuracy on this challenge in the Python code. Here is the input:\n```python\n{code}\n```\n\nNow output a Python script. You should only return straight-line Python code."
            prompt = f"I have a Python script that I would like you to edit:\n```python\n{code}\n```\nAssume you are an experienced data scientist. Please find out if my script is missing crucial steps or has unusual steps compared to the following script examples:\n```python\n{scripts_sampled[0]}\n```\n```python\n{scripts_sampled[1]}\n```\n\n```python\n{scripts_sampled[2]}\n```\n```python\n{scripts_sampled[3]}\n```\n```python\n{scripts_sampled[4]}\n```\nSeveral thoughts on making the script examples shorter:\n1. Select several representative scripts\n2. Remove unnecessary segments (e.g., comments, importing libraries)\n3. Focus on one code module at a time\nNow output a Python script. You should only return straight-line Python code."
            prompt = f"I want to preprocess a dataset from a Kaggle competition, and wrote the following Python script for that:\n```python\n{code}\n```\nHowever, I'm unsure if I'm missing any steps or should leave some steps out. Specifically, I want my Python script above to be similar to what other people are doing. I gave you three scripts of other people below. Please alter my Python program above, and include or exclude steps based on what the other scripts below do. Your output should consist of one python script that compiles and runs, nothing more.:\n```python\n{scripts_sampled[0]}\n```\n```python\n{scripts_sampled[1]}\n```\n\n```python\n{scripts_sampled[2]}\n```\nNow output a Python script. You should only return straight-line Python code."
            
            print(prompt)
            responses = ask_GPT_model(model, system_intel, prompt)
            
            for i, response in enumerate(responses):
                # Use re.findall to find all matches
                matches = re.findall(pattern, response, re.DOTALL)
                try:
                    ouput_script = matches[0].strip()
                    # Print the extracted content
                    print(ouput_script)
                    # Save the extracted content to a file
                    with open(os.path.join(dirctory_save_to, folder, f"{i}.py"), 'w') as f:
                        f.write(ouput_script)
                except: continue