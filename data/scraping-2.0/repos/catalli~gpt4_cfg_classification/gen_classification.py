from openai import OpenAI
import sys
import random
import re

keyfile = "key.txt"

keystr = ""

test_index_max = 11

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

    input_file_path = 'prompt.txt'
    output_file_path = f"output_{test_index}.txt"
    train_file_path = data_prefix+'/train.txt'
    test_file_path = data_prefix+'/test.txt'
    summary_file_path = data_prefix+'/summary.txt'

    pattern_p_desc = ""

    # Read the prompt from the input file
    with open(input_file_path, 'r') as file:
        prompt = file.read()

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

    classifications = []

    no_class = 0
    no_success = 0
   

    print("Natural language description of Pattern P (not given to model):")
    print(pattern_p_desc)
    print("")

    print("Prompt given to model for each classification (with TEST_SENTENCES replaced with the actual sentence to be classified):")
    print(prompt)
    print("")

    for line in shuffled_test_lines:
        gnd_truth = "Label: True" in line
        # Removing the actual label in the version that gets sent to the model
        line = line.replace("Label: True","Label:").replace("Label: False","Label:")

        prompt_input = prompt.replace("TEST_SENTENCES", line)

        completion = generate_completion(client,prompt_input)

        print(completion.replace("\n\n","\n").strip("\n").strip())

        no_class+=1

        classifications.append(completion)

        label = False

        if "Label: True" in completion:
            label = True

        if label == gnd_truth:
            no_success+=1
        
        if no_class !=0:
            print(f"Correct classifications: {no_success}/{no_class}")
            print("")

    if no_class > 0:
        print("Overall success rate: "+str(no_success/no_class))

    with open(output_file_path, 'w') as file:
        file.write("\n".join(classifications))

    print(f"Completions generated and saved to {output_file_path}")

if __name__ == "__main__":
    try:
        test_index = int(sys.argv[1])
    except IndexError:
        test_index = 1
    main(test_index)
