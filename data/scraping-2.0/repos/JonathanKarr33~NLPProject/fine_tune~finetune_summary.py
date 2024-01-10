import os
from openai import OpenAI
import json

api_key = 'YOUR_API_KEY'
client = OpenAI()

def summarize_text(input_text):
    #model = "ft:gpt-3.5-turbo-1106:personal::8UJo3EgI" #! need to change this to the current finetuned model
    model = "ft:gpt-3.5-turbo-1106:personal::8W1NDZLc"
    max_tokens = 4000
    text_to_gpt = f"Make a direct abstract for the following paper: \n{input_text}"
    # Call the OpenAI API to generate text
    response = client.chat.completions.create(model=model,
                                              temperature=.2,
                                              messages=[
                                                  {"role": "user", "content": text_to_gpt}]
                                              )

    # Extract the generated text from the response
    generated_text = response.choices[0].message.content

    return generated_text


def read_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    # Define the file paths
    data_dir = "../data/thousand_papers/" #! change this
    input_file_path = f"{data_dir}//thousand_papers_shrunk2.json"
    output_file_path = f"{data_dir}/our_result.json"

    # Ensure the output directory exists
    os.makedirs(data_dir, exist_ok=True)

    gpt_files = 0
    with open(input_file_path, 'r') as file:
        text = read_file(input_file_path)

    # Generate the summary
    result_list = []
    for article in text[900:]: #! tweak this
        print(gpt_files)
        result_dict = {"article_id": article["article_id"],
                    "abstract_text": article["abstract_text"],
                    "article_text_summary": summarize_text(article["article_text"])}  # Renamed key
        result_list.append(result_dict)

        # Don't waste money
        gpt_files += 1
        if gpt_files >= 100:
            break

    with open(output_file_path, 'w') as file:
        json.dump(result_list, file, indent=2)

    print(f"Summarized and saved {gpt_files} files to {output_file_path}")
