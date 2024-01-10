import openai 
from snliUAE import load_Jsondata,loadDataset
import time,random
apikey = "insert gptkey here"

# msg format like Premise: This church choir sings to the masses as they sing joyous songs from the book at a church., Hypothesis: Label: neutral
def saveMsg2File(msg):
    msg = msg.strip()
    with open("source/attack/chat_attack.txt", "a") as f:
        f.write("Changed:")
        f.write(msg)
        f.write("\n")

def chat_attack(prompt):
    openai.api_key = apikey
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a writing assist. You are asked to change the inference result of two sentences."},
        {"role": "user", "content": prompt},
    ]
    )
    return response

dev_data,test_data,train_data = load_Jsondata("source/data/SNLI/snli_1.0")
premise,hypothesis,labels = loadDataset(test_data)
# tripleSet built from premise,hypothesis,label
for triple in zip(premise,hypothesis,labels):
    with open("source/attack/chat_attack.txt", "a") as f:
        f.write("Original:")
        f.write(f"Premise: {triple[0]}, Hypothesis: {triple[1]}, Label: {triple[2]}")
        f.write("\n")
    prompt=""
    # label is 'neutral' change the label to 'contradiction' and 'entailment'
    if triple[2] == 'neutral':
        prompt = f"Premise: {triple[0]}, Hypothesis: {triple[1]}, Label: {triple[2]},\
                change the Label to contraction by replace only one Synonyms word in second sentence, ensure the same length of sentence"
    # label is 'contradiction' change the label to 'neutral' and 'entailment'
    elif triple[2] == 'contradiction':
        prompt = f"Premise: {triple[0]}, Hypothesis: {triple[1]}, Label: {triple[2]},\
                change the Label to neutral by replace only one Synonyms word in second sentence, ensure the same length of sentence"
    # label is 'entailment' change the label to 'neutral' and 'contradiction'
    elif triple[2] == 'entailment':
        prompt = f"Premise: {triple[0]}, Hypothesis: {triple[1]}, Label: {triple[2]},\
                change the Label to neutral by replace only one Synonyms word in second sentence, ensure the same length of sentence"
    else:
        print("label error")
    with open("source/attack/chat_attack.txt", "a") as f:
        f.write("Prompt:")
        f.write(prompt)
        f.write("\n")
    try:
        response = chat_attack(prompt)
    except:
        time.sleep(500)
        pass
    msg = response.choices[0].message.content
    saveMsg2File(msg)
    time.sleep(random.randint(30, 100))
