import openai

openai.api_key = 'Your API Key'

def read_dataset(filename):
    with open(filename, 'r') as file:
        dataset = [line.strip() for line in file]
    return dataset

def ai_generated_text_detection(text_to_check, dataset):
    prompt = f"Does the following content appear to be written by an AI or chatgpt?\n{text_to_check}\n"

    dataset_prompt = "\n".join(dataset)  # Combine all dataset texts into a single prompt
    prompt += dataset_prompt

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100
    )

    ai_response = response['choices'][0]['text'].strip()

    return ai_response


# Load dataset from 'dataset.txt'
dataset_file = 'AI_Content_Detector\Dataset.txt'
dataset = read_dataset(dataset_file)

text_to_check = input("Enter Text to be Checked: ")
result = ai_generated_text_detection(text_to_check, dataset)
print(result)
