import os
import openai
import pandas as pd
import pkg_resources

# CHECK THE MAIN FUNCTION FOR THE FLOW OF WHAT HAPPENS IN THIS SCRIPT

def parse_data(row):
    """
    Description:
    This function takes a row from the input data and extracts the 'Results' column. It then parses the data into a dictionary format, where keys are emotional categories (Neutral, Enthusiasm, Joy, Hope, Satisfaction, Sad, Anger, Fear), and values are corresponding scores.

    Parameters:
        row: A pandas DataFrame row containing the 'Results' column.
    Returns:
        A dictionary with emotional categories as keys and scores as values.
    """
    
    
    data = row['Results']
    texts = data.split(',')
    texts = [text.replace('[', '').replace(']','').replace('{', '').replace('}', '').replace('=',':').split(':') for text in texts]
    all = {}
    for text in texts:
        all[text[0].strip()] = int(text[1].strip())
    return all


def generate_uuid_list(path_to_output_csv="output.csv"):
    # Make sure there's an output.csv, if there is not, make one.
    try:
        df = pd.read_csv(path_to_output_csv, encoding="utf-8")
    except:
        df = pd.DataFrame(columns=['UUID', 'Clean_Text', 'Results', 'Neutral', 'Enthusiasm', 'Joy', 'Hope', 'Satisfaction', 'Sad', 'Anger', 'Fear'])
        df.to_csv(path_to_output_csv, index=False, encoding="utf-8")

    return df['UUID'].values.tolist()  # List of UUIDs


# params is a list of strings indicating what you would like to select
def prep_data(params, input_file_path):
    df = pd.read_csv(input_file_path, encoding="utf-8", dtype=str)
    df = df.loc[:, params]
    return df


# PACKAGE REQUIREMENT CHECK
def check_req(requirements):
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    end_flag = 0  # Flag to indicate if any package is missing

    for package in requirements:
        if package not in installed_packages:
            print(f"Package '{package}' is missing.")
            end_flag = 1
    for package in requirements:
        if package not in installed_packages:
            print(f"Package '{package}' is initialized.")

    if end_flag:
        print("Download the required packages before you continue.")
        exit
    else:
        print("All packages are successfully downloaded!")


# READING THE CONFIG FILE
def read_config():
    with open("config.txt", "r") as config_file:
        config = {}
        for line in config_file:
            if "=" in line:
                key, value = line.strip().split("=")
                config[key.strip()] = value.strip()
            else:
                print(
                    f"Warning: Skipping line '{line.strip()}' in config file because it does not contain an '=' sign.")
    return config


# READING THE REQUIREMENTS FILE
def read_req():
    with open("requirements.txt", "r") as req_file:
        req = set(line.strip() for line in req_file)
    return req


# PULLING RESULTS FROM OPENAI
def analyze_emotions(api_key, text, prompt, i, retry=0):
    openai.api_key = api_key

    response = None
    try:
        print(f"Starting row {i+1}")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        print(f"Row {i+1} has been completed")
    except Exception as e:
        if retry < 5:
            print(f"Retrying Row {i+1}: {retry}")
            print(f"API request for row '{i+1}' failed. Reason: {str(e)}")
            analyze_emotions(api_key, text, prompt, i, retry+1)
        return f"Error: {str(e)}"

    return response["choices"][0]["message"]["content"]
    # return "Neutral: 7, Enthusiasm: 2, Joy: 1, Hope: 4, Satisfaction: 3, Sad: 1, Anger: 1, Fear: 1"


def run_rows(api_key, data, column_to_read, output_file_path):
    df = data

    emotions_analysis_list = []

    for i, row in df.iterrows():
        journal_entry = row[column_to_read]

        emo = [
            "neutral",
            "enthusiasm",
            "joy",
            "hope",
            "satisfaction",
            "sad",
            "anger",
            "fear",
        ]
        prompt = f"You are a helpful assistant that analyzes emotions. Analyze the text and respond in the following json format [{emo[0]} : score_1, {emo[1]} : score_2, {emo[2]} : score_3, {emo[3]} : score_4, {emo[4]} : score_5, {emo[5]} : score_6, {emo[6]} : score_7, {emo[7]} : score_8]. Match the format EXACTLY, giving a score (min 1 to max 10) for each of the 8 emotions. ONLY analyze these emotions."
        emotions_analysis = analyze_emotions(api_key, journal_entry, prompt, i)
        emotions_analysis_list.append(emotions_analysis)

    df['Results'] = emotions_analysis_list
    df[emo] = df.apply(parse_data, axis=1,result_type='expand')
    return df


def main():
    # GETTING THE DIRECTORY SET
    current_directory = os.getcwd()
    script_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_directory)

    # CHECKS FOR PACKAGES
    requirements = read_req()
    check_req(requirements)

    # LOAD IN CONFIG
    config = read_config()

    api_key = config.get("api_key")
    input_file_path = config.get("input_file_path")
    column_to_read = config.get("column_to_read")  # No need to convert to int here
    uid_to_read = config.get("uid_column")
    output_file_path = config.get("output_file_path")

    # Check for output.csv, create, and export UUIDs
    uuid_list = generate_uuid_list()  # JUST UUIDS

    # Clean input dataset to the two columns
    data = prep_data([uid_to_read, column_to_read], input_file_path)  # UUIDS AND CLEAN TEXT
    
    # Filters out existing uuid from new request
    for uuid in uuid_list:
        data = data.drop(data.index[data['UUID'] == uuid])
    
    # Run the data through GPT
    data = run_rows(api_key, data, column_to_read, output_file_path)


    data.to_csv('output.csv', mode='a+', encoding="utf-8", index=False, header=False)
    print("Done")
    os.chdir(current_directory)


if __name__ == "__main__":
    main()
