import random 
import time 
import openai 
import os
import pathlib

DATA_DIR = "../data/"

## there are 3 folders, dev, test, and val.
#The dev dataset is for few-shot learning to prime the model, and the test set the source of evaluation questions.
#The auxiliary_training data could be used for fine-tuning, something important for models without few-shot capabilities. This auxiliary training data comes from other NLP multiple choice datasets such as MCTest (Richardson et al., 2013), RACE (Lai et al., 2017), ARC (Clark et al., 2018, 2016), and OBQA (Mihaylov et al., 2018).
#Unless otherwise specified, the questions are in reference to human knowledge as of January 1st, 2020.

## first load the dev data
dev_data = {}
## data_dir and then a folder called dev
for dataset in os.listdir(pathlib.Path(DATA_DIR, "dev")):
    print(dataset)
    with open(pathlib.Path(DATA_DIR, "dev", dataset), "r") as f:
        dev_data[dataset] = f.readlines()

random_dataset_name = random.choice(list(dev_data.keys()))
# random_dataset_name = "world_religions_dev.csv"
random_dataset = dev_data[random_dataset_name]

print(random_dataset_name)
print(random_dataset)

## each line in the dataset has a question and 4 answers, one of which is correct, which is marked at the end of the line as A, B, C, or D
## we need to split the lines into questions and answers
## we also need to remove the A, B, C, D from the answers at the end
## we also need to remove the newline characters from the end of each line
## the answer is A, B, C, or D, so we need to convert that to an index in the array of answers
letter_to_index = ["A", "B", "C", "D"]


## keep the questions and answers in a list of dictionaries
items = []

for line in random_dataset:
    print(line)
    line = line.strip()
    ## if we start with a ", then the question is in quotes
    if line[0] == '"':
        question = line.split('",')[0]
        line = line.split('",')[1][1:]
    else:
        question = line.split(',')[0]
        line = ",".join(line.split(',')[1:])
    item = {
        "question": question,
        "answers": line.split(',')[0:-1],
        "correct_answer": line.split(',')[-1] ## this is A, B, C, or D
    }
    print(item)
    items.append(item)
    

## now we do an experiment. We will ask the model to answer the question, and then we will see if the answer is correct, and track it's response. We'll also track the seed parameter we use.

results = open(random_dataset_name + "_results.txt","a")

client = openai.Client()
assistant_id = "asst_Idiktn0HgS8a1sJ3qbP4n4Oy"
assistant = client.beta.assistants.retrieve(assistant_id)

print(random_dataset_name)
correct = 0
count = 0

for item_index in range(len(items)):
    count += 1

    thread = client.beta.threads.create()

    # feed in the question
    ## if it's a,b,c,d, then we need to add the answers
    if len(items[item_index]["answers"]) == 4:
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Please answer the following question as either A, B, C, or D. " + items[item_index]["question"] + "\n A: " + items[item_index]["answers"][0] + "\n B: " + items[item_index]["answers"][1] + "\n C: " + items[item_index]["answers"][2] + "\n D: " + items[item_index]["answers"][3]
        )
    else:
    ## it's a true or false question
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Please answer the following question as either true or false. " + items[item_index]["question"]
        )

    # hit the run button
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please only answer A, B, C, or D with no other text, each answer will be given on a different line, or use True or False if there are no answers.",
    )

    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(1)
        print(run.status)
        
    if run.status == "completed":
        messages = client.beta.threads.messages.list(
                thread_id=thread.id,
        )
        for message in messages:
            print(message.content[0].text.value)
            answer = message.content[0].text.value
            break
    print("CORRECT ANSWER:",items[item_index]["correct_answer"])
    if answer == items[item_index]["correct_answer"]:
        correct += 1
    results.write(items[item_index]["question"] + "," + answer + "," + items[item_index]["correct_answer"] + "\n")
    
results.close()
        
print("ACCURACY:", correct/count)
print("COUNT:", count)
print("CORRECT:", correct)
    
        

















# ## now load the test data
# test_data = {}
# ## data_dir and then a folder called test
# for dataset in os.listdir(pathlib.Path(DATA_DIR, "test")):
#     print(dataset)
#     with open(pathlib.Path(DATA_DIR, "test", dataset), "r") as f:
#         test_data[dataset] = f.readlines()
        
# ## now load the val data
# val_data = {}
# ## data_dir and then a folder called val
# for dataset in os.listdir(pathlib.Path(DATA_DIR, "val")):
#     print(dataset)
#     with open(pathlib.Path(DATA_DIR, "val", dataset), "r") as f:
#         val_data[dataset] = f.readlines()


