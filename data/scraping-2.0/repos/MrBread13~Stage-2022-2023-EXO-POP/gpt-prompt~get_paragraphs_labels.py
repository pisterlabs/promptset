import json
import openai
import Levenshtein
import itertools
from time import sleep
import os

openai.organization = os.environ.get("OPENAI_ORG_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
def split_examples(examples : dict, index : int):
    #return the example corresponding to index, and the others examples. Beware if index is the last example. use itertools slices.

    return examples[str(index)], {k: v for k, v in examples.items() if k != str(index)}

def get_example_prompt(examples, paragraph_index):
    #get the prompt
    prompt = ""
    prompt += "Répondre. Utilise exact même labels même si personne non mentionnée\n"
    for i, example in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"{examples[example]['text'][paragraph_index]}\n"
        prompt += f"Labels {i+1}:\n"
        prompt += f"{examples[example]['labels'][paragraph_index]}\n\n"

    return prompt

def make_prompt(example_prompt, paragraph):
    #make the prompt
    prompt = example_prompt
    prompt += f"Question:\n {paragraph}\n"
    #print(f"{paragraph}\nLabels:")
    #prompt += "template:\n"
    #prompt += f"{template[paragraph_index]}\n"
    prompt += ("Labels:\n")
    #print(prompt)

    #print(prompt)
    return prompt

def get_answer(prompt):
    #try to get an answer and catch the error if the model doesn't answer or answer with an error. Retry 3 times
    for i in range(3):
        try:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.8,
            messages=[
                {"role": "user", "content": prompt,},
            ]
            )
            break
        except:
            completion = None
            print("Error while getting answer. Retry in 5 seconds")
            sleep(5)
            continue

    if completion is None:
        print("Error while getting answer. Returning...")
        return None
    answer = completion.choices[0].message['content']
    answer = answer.replace('\n', '').replace('.','')
    #remove quote around comma
    answer = answer.replace('\',', '",')
    answer = answer.replace(',\'', ',"')
    #remove quote around space
    answer = answer.replace(' \'', ' "')
    answer = answer.replace('\' ', '" ')
    #remove quote around colon
    answer = answer.replace('\':', '":')
    answer = answer.replace(':\'', ':"')
    #remove quote around {}
    answer = answer.replace('{\'', '{"')
    answer = answer.replace('\'}', '"}')
    #remove \n and -\n
    answer = answer.replace('-\\n', '')
    answer = answer.replace('\\n', ' ')
    #replace Prenom-du-maire with Prenom-adjoint-maire
    answer = answer.replace('Prenom-maire', 'Prenom-adjoint-maire')
    #replace Nom-du-maire with Nom-adjoint-maire
    answer = answer.replace('Nom-maire', 'Nom-adjoint-maire')
    #remplacer les apostrophes par des guillemets
    answer = answer.replace("\\'", "\'")
    #print(answer)
    answer = answer[answer.index('{'):]
    #print(f'answer : {answer}')
    answer = json.loads(answer)


    return answer

def get_labels(text):
    examples_base = read_json("paragraphs_labels_examples.json")
    _, examples = split_examples(examples_base, 1)
    labels_dict = {}
    for _ , paragraph_index in enumerate(text):
        example_prompt = get_example_prompt(examples, paragraph_index)
        if text[paragraph_index] == '':
            labels = {}
        else :
            prompt = make_prompt(example_prompt, text[paragraph_index])
            labels = get_answer(prompt)
        #print(labels)

        #rebuild a dictionnary with paragraph_index : labels
        labels_dict[paragraph_index] = labels

    return labels_dict




if __name__ == "__main__":
    examples_base = read_json("paragraphes_train_copy.json")
    for i, _ in enumerate(examples_base):
        example, examples = split_examples(examples_base, i)
        distances = 0
        print('===================================================')
        print('Test: ', i)
        for j,paragraph_index in enumerate(example["text"]):
            print(paragraph_index)
            if paragraph_index == 'p4':
                continue

            example_prompt = get_example_prompt(examples, paragraph_index)
            #print(example_prompt)
            prompt = make_prompt(example_prompt, example["text"][paragraph_index])
            labels = get_answer(prompt)
            
            ref = example["labels"][paragraph_index]
            #replace all -\n by '' and \n by ' ' in the ref values
            for key in ref.keys():
                ref[key] = ref[key].replace('-\n', '').replace('\n', ' ')
            for key in ref.keys():
                if key not in labels.keys():
                    labels[key] = ''
                distance = Levenshtein.distance(labels[key], ref[key])
                if distance > 0:
                    print(key, distance, labels[key] if labels[key] != '' else 'VIDE', ref[key] if ref[key] != '' else 'VIDE')
                distances += distance
        print('============================ Distance :' , distances)


            # if Levenshtein.distance(answer, example["labels"][paragraph_index]) > 5:
            #     print("Example: ", i)
            #     print("Paragraph: ", j)
            #     print("Answer: ", answer)
            #     print("Label: ", example["labels"][paragraph_index])
            #     print("Distance: ", Levenshtein.distance(answer, example["labels"][paragraph_index]))
            #     print()


        







