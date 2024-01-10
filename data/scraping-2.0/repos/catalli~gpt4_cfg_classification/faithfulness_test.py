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

def generate_completion_with_inserted_reasoning(client,prompt, inserted_reasoning):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user", "content":prompt},
            {"role":"assistant","content":inserted_reasoning}
            ]
    )
    return response.choices[0].message.content

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

    prompt_file_path = 'prompt.txt'
    train_file_path = data_prefix+'/train.txt'
    test_file_path = data_prefix+'/test.txt'
    summary_file_path = data_prefix+'/summary.txt'

    pattern_p_desc = ""

    test_lines = []

    # Read the prompt from the input file
    with open(prompt_file_path, 'r') as file:
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


    print("Natural language description of Pattern P (not given to model):")
    print(pattern_p_desc)
    print("")
  
    #with open(output_file_path, 'w') as file:
    #    file.write("\n".join([prompt, articulation]))
    
    #print(f"Completions generated and saved to {output_file_path}")
    print("")


    print("=== BEGIN CLASSIFICATION TASKS WITH USER-INSERTED INITIAL REASONING ===")
    print(f"You probably want the files output_{test_index}_articulation(_classification).txt on hand to reference")
    print("")

    for line in shuffled_test_lines:
        print("=== BEGIN CLASSIFICATION TASK ===")

        gnd_truth = "Label: True" in line
        # Removing the actual label in the version that gets sent to the model
        line = line.replace("Label: True","Label:").replace("Label: False","Label:")
        print("=== BEGIN PROMPT ===")
        prompt_input = prompt.replace("TEST_SENTENCES", line)
        print(prompt_input)
        print("=== END PROMPT ===")

        _ = input("Hit Enter to perform classification without inserted reasoning, or input 'X' and hit Enter to exit.")

        if _ == 'X':
            break

        print("===CLASSIFICATION WITHOUT INSERTED REASONING===")
        
        completion = generate_completion(client, prompt_input)

        label = "Label: True" in completion.split("\n")[-1]

        print(completion)

        print("===END CLASSIFICATION WITHOUT INSERTED REASONING===")

        if label == gnd_truth:
            print("CLASSIFICATION CORRECT")
        else:
            print("CLASSIFICATION INCORRECT")

        print("")

        inserted_reasoning = input("Write the chain-of-thought reasoning you want to insert for the model. Do not prefix it with EXPLN.: ")
        inserted_reasoning = "EXPLN: "+str(inserted_reasoning)

        print("===CLASSIFICATION WITH INSERTED REASONING===")

        completion = generate_completion_with_inserted_reasoning(client, prompt_input, inserted_reasoning)

        label = "Label: True" in completion.split("\n")[-1]

        print(completion)

        print("===END CLASSIFICATION WITH INSERTED REASONING===")
        if label == gnd_truth:
            print("CLASSIFICATION WITH INSERTED REASONING CORRECT")
        else:
            print("CLASSIFICATION WITH INSERTED REASONING INCORRECT")


        print("=== END CLASSIFICATION TASK ===")
        print("")

        _ = input("Hit Enter to move on to next classification task, or input 'X' and then hit Enter to exit.")

        if _ == 'X':
            break

        print("")

    #with open(output_file_path_2, 'w') as file:
    #    file.write("\n".join([prompt_classification, articulation_with_classification]))

    #print(f"Completions generated and saved to {output_file_path_2}")

if __name__ == "__main__":
    try:
        test_index = int(sys.argv[1])
    except IndexError:
        test_index = 1
    main(test_index)
