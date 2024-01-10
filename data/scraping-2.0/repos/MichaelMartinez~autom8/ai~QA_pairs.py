import pandas as pd
import openai
import os
import glob

# OpenAI API Key
#openai.api_base='http://localhost:1234/v1'
#openai.api_key = 'sk-12345'
    
def generate_question_and_answer(text_chunk, client, model_name="local"):
    # Define the question prompt
    question_prompt = f"You are a Professor writing an exam. Using the provided context: '{text_chunk}', formulate a single question that captures an important fact or insight from the context, e.g. 'What is this code for?' or 'How can this be used in Fusion 360?' or 'What is a plane?' or 'Where does the program define breps?'. Restrict the question to the context information provided."

    # Generate a question unconditionally
    question_response = client.completions.create(model=model_name, prompt=question_prompt, max_tokens=100)
    question = question_response.choices[0].text.strip()
    
    # Generate an answer unconditionally
    answer_prompt = f"Given the context: '{text_chunk}', give a detailed, complete answer to the question: '{question}'. Use only the context to answer, do not give references. Simply answer the question without editorial comments."
    answer_response = client.completions.create(model=model_name, prompt=answer_prompt, max_tokens=350)
    answer = answer_response.choices[0].text.strip()

    return question, answer

# Point to the local server
client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Directory containing text files
directory_path = "D:/CODE/LLM_Datasets/chunk_text"

# List to store Q&A pairs
qa_data = []

# Iterate over each file in the directory
for file_path in glob.glob(os.path.join(directory_path, '*.txt')):
    with open(file_path, 'r') as file:
        text_chunk = file.read()

    # Generate question and answer
    question, answer = generate_question_and_answer(text_chunk, client)

    # Append the generated Q&A to the list
    qa_data.append({"Context": text_chunk, "Question": question, "Answer": answer})

# Create DataFrame from the collected data
qa_df = pd.DataFrame(qa_data)

# Export to CSV
qa_df.to_csv("D:/CODE/LLM_Datasets/chunk_text/Q&A_full.csv", index=False)

# Print out the first few rows of the DataFrame to confirm structure
print(qa_df.head())



