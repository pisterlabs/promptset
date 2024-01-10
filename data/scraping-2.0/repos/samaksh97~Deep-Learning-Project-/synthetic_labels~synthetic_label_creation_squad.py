from openai import OpenAI
import pandas as pd
from datasets import load_dataset
import json
import sys
from collections import defaultdict
from dotenv import dotenv_values
from squad_synthetic_data_process import clean_data

config = dotenv_values('../.env.local')

client = OpenAI(
  api_key = config['OPENAI_API_KEY'],
)  

# Question: What is the Grotto at Notre Dame?
# Answer: a Marian place of prayer and reflection
# Question: What sits on top of the Main Building at Notre Dame?
# Answer: a golden statue of the Virgin Mary

def preprocess_data(dataset):
    context_dict = {}
    question_dict = defaultdict(list)
    answer_dict = defaultdict(list)
    
    start = 0
    
    for end in range(len(dataset['question'])):
        batch_questions = []
        batch_answers = []
        if end ==0:
            continue
        if end == len(dataset['question'])-1 or dataset["context"][end-1] != dataset["context"][end]:
            batch_questions.extend(dataset['question'][start:end])
            batch_answers.extend(dataset["answers"][start:end])
            context_dict[len(context_dict.keys())] = dataset["context"][start]
            batch_answers = [i['text'][0] for i in batch_answers]
            question_dict[len(question_dict.keys())] = batch_questions
            answer_dict[len(answer_dict.keys())] = batch_answers
            start = end
        
    return context_dict, question_dict, answer_dict
    
def generate_answer(ques_str, context):
    
    messages=[
                {"role": "system", "content": '''You are an assistant which can search for answers within a reading passage (context) for a given question. 
                    The answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. 
                    Following are a few examples of answers. Generate the entity only and not the complete sentence in the answer for the given questions and context and return answers in a numbered list.
                    Question: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
                    Answer: Saint Bernadette Soubirous
                    Question: What is in front of the Notre Dame Main Building?
                    Answer: a copper statue of Christ
                    Question: The Basilica of the Sacred heart at Notre Dame is beside to which structure?
                    Answer: the Main Building
                    '''},
                {"role": "user", "content": f"Context: {context}\n{ques_str}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    # Extracting the answer from the GPT-3 response
    answer = response.choices[0].message.content
    return answer

def main(context_dict, questions_dict):
    synthetic_dict = defaultdict(list)
    
    # Example usage with a few questions from SQuAD
    for i in range(len(context_dict.keys())):
        context = context_dict[i]
        
        ques_str = ''
        for ques in questions_dict[i]:
            ques_str += f'Question: {ques}\n'
        
        # Generate a synthetic answer
        synthetic_str = generate_answer(ques_str, context)
        print(f"Context --> {i+1} processed\n")

        synthetic_dict[i] = synthetic_str

        # Save intermediate results in json format to avoid losing progress
        if i%10 == 0 or i == len(context_dict.keys())-1:
            with open('../data/synthetic_answers_squad_validation.txt', 'w') as json_file:
                json.dump(synthetic_dict, json_file)
                
    return synthetic_dict
            
    

if __name__ == '__main__':

    RUNNING_KEY = 'validation' ## set to "training" or "validation"

    # Load the SQuAD dataset
    squad_dataset = load_dataset("squad")

    # Access the training and validation sets
    dataset = squad_dataset[RUNNING_KEY][:2520]

    context_dict, questions_dict, answers_dict = preprocess_data(dataset)

    print('Total context found -->', len(context_dict.keys()))

    synthetic_dict = main(context_dict, questions_dict)

    cleaned_synthetic = clean_data(synthetic_dict, f'squad_synthetic_processed_{RUNNING_KEY}')

    # If using a saved file
    # with open('../data/synthetic_processed_validation.json', 'r', encoding='utf-8') as f:
    #     cleaned_synthetic = json.load(f)

    temp = []
    for key in context_dict.keys():
        temp.append({
            'context': context_dict[key],
            'records': [{'question': q, 'answer': a, 'synthetic_answer': sa} for (q, a, sa) in list(zip(questions_dict[key], 
                                                                                                        answers_dict[key], 
                                                                                                        cleaned_synthetic[key]))]
        })
    
    ## Saving combined data
    with open(f'../data/squad_original_{RUNNING_KEY}.json', 'w', encoding='utf-8') as f:
        json.dump(temp, f, ensure_ascii=False, indent=4)

    # ## Creating dataframe for original data
    # with open('data_original.json','r') as f:
    #     data = json.loads(f.read())
        
    ## Normalizing data
    df = pd.json_normalize(temp, record_path=['records'], meta=['context'])
    
    df.to_csv(f'../data/squad_df_{RUNNING_KEY}.csv', index=False)
