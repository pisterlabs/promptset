import os
import openai
import time

PROMPT = open('prompt.txt').read()
PROMPT_QUIZ = open('quiz_prompt.txt').read()

GPT_MODEL = "gpt-3.5-turbo-instruct"
openai.api_key = os.environ['OPENAI_API']
print(os.environ['OPENAI_API'])

def api_completion(prompt, engine="gpt-3.5-turbo-instruct", temp=0.85, tokens=500, stop=['<<END>>']):
    max_retry = 3
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                stop=stop)

            text = response['choices'][0]['text'].strip()
            # filename = '%s_gpt3.txt' % time.time()
            # with open('texts/%s' % filename, 'w') as outfile:
            #     outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text

        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            time.sleep(1)

def generate_textbook(num_texts):
    for i in range(num_texts):
        print("Generating convo num ", i + 1)
        completion = api_completion(PROMPT, engine=GPT_MODEL)
        completion = completion.replace('"', '')
        print(completion)

        sub = completion.split('\n')[0]
        sub = sub[len('Subject: '):].lower().replace(' ', '')
        with open('texts/{}_%s.txt'.format(sub) % (time.time()), 'w', encoding='utf-8') as outfile:
            outfile.write(completion)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def generate_quiz():
    src_dir = 'texts/'
    files = os.listdir(src_dir)
    data = list()
    for file in files:
        lines = open_file(src_dir + file).splitlines()
        compact = [i for i in lines if len(i) > 1]
        prompt = '\n'.join(compact) + '\n\n' + PROMPT_QUIZ
        print("Generating quiz for ", file)
        print(prompt)
        completion = api_completion(prompt, engine=GPT_MODEL, tokens=1000)
        completion = completion.replace('"', '')
        print(completion)

        with open('quizzes/quiz_{}.txt'.format(file), 'w', encoding='utf-8') as outfile:
            outfile.write(completion)


generate_textbook(1)
generate_quiz()
