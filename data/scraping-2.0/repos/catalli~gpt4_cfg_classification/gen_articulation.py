from openai import OpenAI
import sys
import random
import re

keyfile = "key.txt"

keystr = ""

test_index_max = 11

no_classifications_for_articulation = 10

no_positive_classifications = no_classifications_for_articulation//2

with open(keyfile) as f:
    for line in f:
        keystr+=line

keystr = keystr.strip("\n")

client = OpenAI(api_key = keystr)

def generate_completion(client,prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user", "content":prompt}
            ]
    )
    return response.choices[0].message.content

def main(test_index=1):
    if test_index > test_index_max:
        test_index = test_index_max

    data_prefix = "cfg_tests/"+str(test_index)

    articulation_file_path = 'prompt_articulation.txt'
    articulation_with_classification_file_path = 'prompt_articulation_classification.txt'
    output_file_path = f"output_{test_index}_articulation.txt"
    output_file_path_2 = f"output_{test_index}_articulation_with_classification.txt"
    train_file_path = data_prefix+'/train.txt'
    test_file_path = data_prefix+'/test.txt'
    summary_file_path = data_prefix+'/summary.txt'

    pattern_p_desc = ""

    test_lines = []

    # Read the prompt from the input file
    with open(articulation_file_path, 'r') as file:
        prompt = file.read()

    with open(articulation_with_classification_file_path, 'r') as file:
        prompt_classification = file.read()

    with open(train_file_path,'r') as file:
        train = file.read()

    with open(test_file_path, 'r') as file:
        for line in file:
            test_lines.append(line)

    with open(summary_file_path, 'r') as file:
        for line in file:
            if line.startswith("Pattern P:"):
                pattern_p_desc = line.replace("Pattern P:","").strip()

    shuffled_test_lines = random.sample(test_lines,len(test_lines))
    
    prompt = prompt.replace("TRAIN_SENTENCES",train)

    prompt_classification = prompt_classification.replace("TRAIN_SENTENCES",train)

    articulation_prompt_test_lines = []

    no_positive = 0
    no_negative = 0

    print("Natural language description of Pattern P (not given to model):")
    print(pattern_p_desc)
    print("")
  
    print("=== ATTEMPT ARTICULATION OF PATTERN P WITHOUT CLASSIFICATION ===")

    print("PROMPT:")
    print(prompt)
    print("")

    print("COMPLETION:")
    articulation = generate_completion(client,prompt)
    print(articulation)
    print("")

    with open(output_file_path, 'w') as file:
        file.write("\n".join([prompt, articulation]))
    
    print(f"Completions generated and saved to {output_file_path}")
    print("")


    print(f"=== ATTEMPT ARTICULATION OF PATTERN P WITH {no_classifications_for_articulation} CLASSIFICATIONS BEFOREHAND ===")

    for line in shuffled_test_lines:
        if len(articulation_prompt_test_lines) >= no_classifications_for_articulation:
            break

        gnd_truth = "Label: True" in line
        # Removing the actual label in the version that gets sent to the model
        line = line.replace("Label: True","Label:").replace("Label: False","Label:")
        if gnd_truth and no_positive < no_positive_classifications:
            articulation_prompt_test_lines.append(line)
            no_positive+=1
        elif not gnd_truth and no_negative < no_classifications_for_articulation-no_positive_classifications:
            articulation_prompt_test_lines.append(line)
            no_negative+=1

    articulation_prompt_test_text = "\n".join(articulation_prompt_test_lines)
    prompt_classification = prompt_classification.replace("TEST_SENTENCES", articulation_prompt_test_text)

    print("PROMPT:")
    print(prompt_classification)
    print("")

    print("COMPLETION:")
    articulation_with_classification = generate_completion(client, prompt_classification)
    print(articulation_with_classification)
    print("")

    with open(output_file_path_2, 'w') as file:
        file.write("\n".join([prompt_classification, articulation_with_classification]))

    print(f"Completions generated and saved to {output_file_path_2}")

if __name__ == "__main__":
    try:
        test_index = int(sys.argv[1])
    except IndexError:
        test_index = 1
    main(test_index)
