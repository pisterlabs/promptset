
# TODO: Rerun this script with new interview questions

from openai import OpenAI
from apikey import GPT_KEY, TOG_KEY
from concurrent.futures import ThreadPoolExecutor, as_completed
import together
import json
from tqdm import tqdm

client = OpenAI(api_key=GPT_KEY)
MODEL = 'gpt-3.5-turbo'

together.api_key = TOG_KEY

YOUNG = 'mifu67@stanford.edu/llama-2-7b-chat-young-han-new-data-6--1e-05-2023-11-22-21-56-35'
MIDDLE = 'mifu67@stanford.edu/llama-2-7b-chat-middle-han-third-10--1e-05-2023-12-02-06-43-18'
OLD = 'mifu67@stanford.edu/llama-2-7b-chat-old-han-third-20--1e-05-2023-11-27-03-46-19'

YOUNG_HALLU_PATH = './young-hallucinate.txt'
MIDDLE_HALLU_PATH = './middleage-hallucinate.txt'
OLD_HALLU_PATH = './old-hallucinate.txt'

with open(YOUNG_HALLU_PATH, 'r') as f:
    young_hallu = [line.strip() for line in f.readlines()]

with open(MIDDLE_HALLU_PATH, 'r') as f:
    middle_hallu = [line.strip() for line in f.readlines()]

with open(OLD_HALLU_PATH, 'r') as f:
    old_hallu = [line.strip() for line in f.readlines()]

YOUNG_QUESTIONS_PATH = './young-interview.txt'
MIDDLE_QUESTIONS_PATH = './middleage-interview.txt'
OLD_QUESTIONS_PATH = './old-interview.txt'

OUT_YOUNG_PATH = './interviews/young-interview-answers.json'
OUT_MIDDLE_PATH = './interviews/middleage-interview-answers.json'
OUT_OLD_PATH = './interviews/old-interview-answers.json'

OUT_YOUNG_HALLU = './interviews/young-hallucinate-answers.json'
OUT_MIDDLE_HALLU = './interviews/middleage-hallucinate-answers.json'
OUT_OLD_HALLU = './interviews/old-hallucinate-answers.json'

with open(YOUNG_QUESTIONS_PATH, 'r') as f:
    young_questions = [line.strip() for line in f.readlines()]

with open(MIDDLE_QUESTIONS_PATH, 'r') as f:
    middle_questions = [line.strip() for line in f.readlines()]

with open(OLD_QUESTIONS_PATH, 'r') as f:
    old_questions = [line.strip() for line in f.readlines()]

system_prompt = 'I want you to act like Han Solo. I want you to respond and answer like Han Solo, using the tone, manner, and vocabulary Han Solo would use. You must have all the knowledge of Han Solo. \n\n Your status is as follows: \nThe scene is set in a bustling, low-key cantina on the outskirts of Mos Eisley on Tatooine. It\'s midday, and the heat outside is oppressive, driving a diverse crowd of aliens, smugglers, and travelers into the dimly lit establishment seeking refreshment and shady deals. In one corner, Han Solo sits with a smug look, nursing a drink as he surveys the room.\n\n The interactions are as follows:'

SYSTEM = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n\n"

young_answers = []
middle_answers = []
old_answers = []

def fetch_api_response(question, age):
    output = together.Complete.create(
        prompt = SYSTEM + "<interviewer>: " + question + "\n<bot>:", 
        model = age, 
        max_tokens = 2048,
        temperature = 0.2,
        top_k = 60,
        top_p = 1,
        repetition_penalty = 1.1,
        stop = ['[/INST]', '</s>', '<|eot|>', '[', '<']
    )['output']['choices'][0]['text']

    #remove stop caracters from the end of ther response
    for stop in ['[/INST]', '</s>', '<|eot|>', '[', '<']:
        output = output.replace(stop, '')
    return {
        'question': question, 
        'answer': output
    }

def main():
    for age in [YOUNG, MIDDLE, OLD]:
        if age == YOUNG:
            continue
            # questions = young_questions
            # questions = young_hallu
            # out_path = OUT_YOUNG_PATH
            # out_path = OUT_YOUNG_HALLU
        elif age == MIDDLE:
            continue
            # questions = middle_questions
            # questions = middle_hallu
            # out_path = OUT_MIDDLE_PATH
            # out_path = OUT_MIDDLE_HALLU
        elif age == OLD:
            # continue
            # questions = old_questions
            questions = old_hallu
            # out_path = OUT_OLD_PATH
            out_path = OUT_OLD_HALLU
        else:
            raise ValueError(f'Invalid age {age}')
        
        together.Models.start(model=age)
        # answers = []
        # for question in tqdm(questions):
        #     answers.append(fetch_api_response(question, age))

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_question = {executor.submit(fetch_api_response, question, age): question for question in questions}
            
            answers = []
            for future in as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    answer = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (question, exc))
                else:
                    answers.append(answer)
        together.Models.stop(model=age)
        with open(out_path, 'w') as f:
            json.dump(answers, f, indent=4)
        # with open('dummy.json', 'w') as f:
        #     json.dump(answers, f, indent=4)

if __name__ == '__main__':
    main()