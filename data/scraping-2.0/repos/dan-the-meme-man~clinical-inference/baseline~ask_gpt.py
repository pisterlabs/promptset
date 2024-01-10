import sys
sys.path.append('.')

import os
import time

import openai

# pulls dataset splits
from data.retrieve_data import get_data

# Set your OpenAI API key
try:
    openai.api_key = open(os.path.join('baseline', 'key.txt'), 'r').read().strip()
except:
    raise ValueError('You must provide a valid OpenAI API key in a file called key.txt in the baseline directory.')

def main():
    
    # pull dev set
    try:
        dev_data = get_data('dev')
    except:
        raise ValueError('Could not load dev set - ensure you have run fetch_task.py and serialize_cts.py.')

    # build prompts and store labels as well
    prompts = []
    labels = []
    for i in range(len(dev_data)):

        prompt = "Given the following information: \n\n"
        
        prompt += dev_data[i].concat_context()
                
        prompt += '\n\nDecide whether the following statement is an entailment or a contradiction: '
        
        prompt += dev_data[i].statement
        
        prompt += '\n\nPlease be sure to use only one of those two words in your response.'
        
        prompts.append(prompt)
        
        if dev_data[i].label == 0:
            labels.append('entailment')
        elif dev_data[i].label == 1:
            labels.append('contradiction')
        else:
            raise ValueError('Label must be 0 or 1.')
        
        print(prompt)
        exit()

    # Split results into true positive, false positive, true negative, and false negative
    tp = []
    fp = []
    tn = []
    fn = []

    # Call the OpenAI API to generate a response for each prompt
    for i in range(len(prompts)):
        response = openai.Completion.create(
            engine="text-davinci-002",  # Use the GPT-3.5 model
            prompt=prompt,
            max_tokens=50  # Limit the total tokens generated to 50
        )

        # Extract the generated text from the response
        generated_text = response.choices[0].text
        
        # decide TP, FP, TN, or FN
        if 'entailment' in generated_text.lower():
            if labels[i] == 'entailment':
                tp.append(generated_text)
            else:
                fp.append(generated_text)
        elif 'contradiction' in generated_text.lower():
            if labels[i] == 'contradiction':
                tn.append(generated_text)
            else:
                fn.append(generated_text)
        
        # avoid being denied due to rate limiting, may need to adjust
        time.sleep(3)
        
    # calculate metrics
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * (prec * rec) / (prec + rec)
    acc = tp + tn / (tp + tn + fp + fn)
    
    # report stats
    with open('ask_gpt.txt', 'w+') as f:
        f.write(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n\n')
        f.write(f'Precision: {prec}, Recall: {rec}, F1: {f1}, Acc: {acc}')

if __name__ == '__main__':
    main()
