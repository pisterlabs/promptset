import openai
import os
from dotenv import load_dotenv
import pandas as pd
import threading

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")


def explain_medical_term(medical_term, model):  # gpt-3.5-turbo-1106
    instruction = """you are a medical professor. you are explaining a medical term to medical students."""
    prompt = f"""explain the medical term '{medical_term}' to medical students. if the term is not correct, write 'not correct' and press enter."""

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=500,
    )

    # 필터링된 단어만 추출
    return response.choices[0].message["content"]


def get_description_with_timeout(medical_term, model, timeout_seconds=30):
    description = "Connecting too long"
    stop_thread = threading.Event()

    def target():
        nonlocal description
        try:
            description = explain_medical_term(medical_term, model).strip()
        except Exception as e:
            if not stop_thread.is_set():
                description = f"Error: {e}"

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        stop_thread.set()
        thread.join()  # Wait for the thread to finish

    return description


def convert_to_csv(basename):
    ngram_data = []
    for n in range(1, 4):
        with open(f'{basename}_{n}_grams.txt', 'r', encoding='utf-8') as file:
            ngram_data.extend([line.split(',')[0].strip() for line in file.readlines()])

    df = pd.DataFrame(ngram_data, columns=['Keyword'])
    df.to_csv(f'{basename}_ngrams.csv', index=False)


def update_with_gpt_descriptions(input_csv_path, reference_csv_path='REFERENCE.csv', model="gpt-3.5-turbo"):
    df = pd.read_csv(input_csv_path)
    reference_df = pd.read_csv(reference_csv_path)
    reference_dict = dict(zip(reference_df['Keyword'], reference_df['Description']))

    if 'Description' not in df.columns:
        df['Description'] = None

    total_rows = len(df)
    for index, row in df.iterrows():
        keyword = row['Keyword']
        if pd.isna(row['Description']):
            if keyword in reference_dict:
                description = reference_dict[keyword]
                action_taken = "updated from reference"
            else:
                description = get_description_with_timeout(keyword, model) + '..'
                action_taken = "updated with GPT description"
            df.at[index, 'Description'] = description
        else:
            action_taken = "already had a description"

        print(f"Row {index + 1}/{total_rows}: {keyword} - {action_taken}")
        df.to_csv(input_csv_path, index=False)

    print("All updates complete.")


def remove_non_medical_terms(input_csv_path, phrases=["not a medical term", "not specific to a medical term",
                                                      "not a specific medical term", "not a correct medical term",
                                                      "not a correct term", "Not correct", "not separate terms",
                                                      "not a recognized medical term"]):
    df = pd.read_csv(input_csv_path)
    # 'Description' 열을 소문자로 변경
    df['Description'] = df['Description'].str.lower()

    # 주어진 모든 문구에 대해 반복하며 해당 문구를 포함하는 행을 제거
    for phrase in phrases:
        df = df[~df['Description'].str.contains(phrase.lower(), na=False)]

    df.to_csv(input_csv_path, index=False)
    print("Non-medical terms removed.")


def append_to_reference(input_csv_path, reference_csv_path):
    # Load the input CSV and reference CSV
    input_df = pd.read_csv(input_csv_path)
    reference_df = pd.read_csv(reference_csv_path)

    # Assuming the input CSV has 'Keyword' and 'Description' columns
    # Filter out rows where 'Description' is not null (i.e., new or updated descriptions)
    updated_rows = input_df[input_df['Description'].notna()]

    # Append these rows to the reference dataframe
    updated_reference_df = pd.concat([reference_df, updated_rows])

    # Removing possible duplicates - keeping the last (most recent) entry
    updated_reference_df.drop_duplicates(subset='Keyword', keep='last', inplace=True)

    # Save the updated reference dataframe back to the reference CSV file
    updated_reference_df.to_csv(reference_csv_path, index=False)

    print("Reference file updated successfully.")
