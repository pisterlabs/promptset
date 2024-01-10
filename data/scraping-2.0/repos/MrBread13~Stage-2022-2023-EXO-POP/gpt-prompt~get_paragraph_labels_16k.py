import json
import openai
import Levenshtein
import itertools
from time import sleep
import os

# openai.organization = os.environ.get("OPENAI_ORG_KEY")
# openai.api_key = os.environ.get("OPENAI_API_KEY")


openai.organization = "org-2wXrLf4fLEfdyawavmkAqi8z"
openai.api_key = "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"

def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
def split_examples(examples : dict, index : int):
    #return the example corresponding to index, and the others examples. Beware if index is the last example. use itertools slices.

    return examples[str(index)], {k: v for k, v in examples.items() if k != str(index)}

def get_example_prompt(examples, paragraph_index):
    prompt = [{"role": "system", "content": "Read the French Mariage Acts input by the user, then answer using a JSON to extract named entities in the act. Always use the same JSON keys. Beware of plurals. Parents can have the same job. They can also live with their child ('avec ses père et mère', 'avec sa mère', 'avec son père'). Do not answer with anything else that what is in the text. Pay attention to cities, departments and countries."}]
    for i, example in enumerate(examples):
        prompt.append({"role": "user", "content": str(examples[example]['text'][paragraph_index])})
        #print(examples[example]['labels'])
        prompt.append({"role": "assistant", "content": str(examples[example]['labels'][paragraph_index])})

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
    print("Getting answer...")
    for i in range(3):
        try:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature=0.4,
            messages=prompt
            )
            break
        except:
            completion = None
            print("Error while getting answer. Retry in 5 seconds")
            sleep(5)
            continue

    if completion is None:
        print("Error while getting answer. Returning...")
        return {}
    answer = completion.choices[0].message['content']
    #print("Raw answer : ", answer)
    answer = answer.replace('\n', '').replace('.','')


    # #remove quote around comma
    # answer = answer.replace('\',', '",')
    # answer = answer.replace(',\'', ',"')
    # #remove quote around space
    # answer = answer.replace(' \'', ' "')
    # #answer = answer.replace('\' ', '" ')
    # #remove quote around colon
    # answer = answer.replace('\':', '":')
    # answer = answer.replace(':\'', ':"')
    # #remove quote around {}
    # answer = answer.replace('{\'', '{"')
    # answer = answer.replace('\'}', '"}')
    #remove \n and -\n


    answer = answer.replace('-\\n', '')
    answer = answer.replace('\\n', ' ')

    answer = answer.replace('"', '\'')

    answer = answer.replace('{\'', '{"')
    answer = answer.replace('{ \'', '{ "')

    answer = answer.replace('\'}', '"}')
    answer = answer.replace('\' }', '" }')

    answer = answer.replace('\':', '":')
    answer = answer.replace('\' :', '" :')

    answer = answer.replace(':\'', ':"')
    answer = answer.replace(': \'', ': "')

    answer = answer.replace('\',', '",')
    answer = answer.replace('\' ,', '" ,')

    answer = answer.replace(',\'', ',"')
    answer = answer.replace(', \'', ', "')


    #replace Prenom-du-maire with Prenom-adjoint-maire
    answer = answer.replace('Prenom-maire', 'Prenom-adjoint-maire')
    #replace Nom-du-maire with Nom-adjoint-maire
    answer = answer.replace('Nom-maire', 'Nom-adjoint-maire')
    #remplacer les apostrophes par des guillemets
    answer = answer.replace("\\'", "\'")
    #print(answer)
    answer = answer[answer.index('{'):]
    print(f'answer : {answer}')
    answer = json.loads(answer)
    #print("Answer : ", answer)

    return answer

def get_labels(text):
    examples = read_json("paragraphs_labels_examples.json")
    labels_dict = {}
    for _ , paragraph_index in enumerate(text):
        print(text[paragraph_index])
        prompt = get_example_prompt(examples, paragraph_index)
        if text[paragraph_index] == '':
            labels = {}
        else :
            prompt.append({"role": "user", "content": f"{text[paragraph_index]}"})
            print(f"==========================Paragraph {paragraph_index}==========================")
            #print("Prompt : ", prompt)
            labels = get_answer(prompt)
        #print(labels)

        #rebuild a dictionnary with paragraph_index : labels
        labels_dict[paragraph_index] = labels

    return labels_dict



        







