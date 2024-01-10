import json
import copy
import openai

# Load your API key from an environment variable or secret management service
api_key = '<your OpenAI API key>'
openai.api_key = api_key

prompt_base = '''Given a sentence that describes a set of visual concepts, these concepts may be Adjectives, Nouns, Verbs, or Adverbs. Please identify these concepts and organize them into a Python list.
'''

caption_path = 'instructBLIP.json'
save_path = 'instructBLIP_gpt4_concepts.json'

def main():
    with open(caption_path, 'r') as f:
        bongard_ow = json.load(f)

        for sample in bongard_ow:
            captions = sample['captions']
            concepts = []
            for caption in captions:
                prompt = copy.deepcopy(prompt_base)
                prompt += 'sentence: ' + caption.strip('.') + '.\n'
                prompt += 'concepts:'
                print(prompt)

                try:
                    response = openai.ChatCompletion.create(model="gpt-4",
                                                            messages=[{
                                                                "role": "user",
                                                                "content": prompt
                                                            }],
                                                            max_tokens=1024,
                                                            temperature=1,
                                                            n=1,
                                                            frequency_penalty=0,
                                                            presence_penalty=0)
                    print(response, '\n')
                    concept_list = response['choices'][0]['message']['content'].replace("\\", '').replace("'", '').replace('"', '').replace("[", '').replace("]", '')
                    concepts.append(copy.deepcopy(concept_list))
                except Exception as e:
                    print(f'response eorror: {e}')
                    concepts.append(copy.deepcopy([concept.strip(', .') for concept in caption.split(' ')]))

            sample['concepts'] = copy.deepcopy(concepts)
            with open(save_path, "w") as file:
                json.dump(bongard_ow, file, indent=4)

if __name__ == '__main__':
    main()