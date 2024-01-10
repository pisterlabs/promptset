import os
import openai
from principles_finder_prompt import FINDER_PROMPT
from will_prompt import PROMPT as BASE_PROMPT
from passage_ranker import load_passage_ranker
import difflib
import json

openai.api_key = os.getenv("OPENAI_API_KEY")
RANKER = load_passage_ranker()
DEBUG = True


def get_principles_to_chapter():
    with open('principles_to_chapter.json', 'r') as f:
        out = json.load(f)
    return out


def find_relevant_chapter(principles, corpus):
    closest_principles = []
    if principles is not None:
        out = []
        for idx, principle in enumerate(principles):
            closest_principle = difflib.get_close_matches(principle, list(corpus.keys()))[0]
            closest_principles.append(closest_principle)
            out.append(''.join(corpus[closest_principle]))
        out = '\n'.join(out)
    else:
        closest_principles = None
        out = None
    return closest_principles, out


def _format_principles_finder_input(questions):
    out = FINDER_PROMPT + '\n'
    if len(questions) == 1:
        return f'{FINDER_PROMPT}\nQuestion: {questions[0]}\n\nAnswer:\n'
    else:
        for idx, question in enumerate(questions):
            if idx != len(questions)-1:
                out = out + f'Question: {question.strip()}\nResponse: ...\n'
            else:
                out = out + f'Question: {question.strip()}\n\nAnswer:\n'
        return out


def find_principles(questions, ranker, corpus):
    # model_input = '; '.join(questions[-2:])
    model_input = questions[-1]
    result = ranker.search(model_input)
    input_result = [i['text'].split('\n')[0] for i in result]
    if DEBUG:
        print('\n' + '#'*5 + 'Principles Finder Result' + '#'*5 + '\n')
        print('Input: ', model_input)
        print('Result: ', result)
        for r in result:
            print('\n')
            print(r['text'].split('\n')[0])
            print(f"score: {r['score']}")
        print('#'*20)
    if result[0]['score'] < -5:
        return None, None
    principles, chapter = find_relevant_chapter(input_result[:2], corpus)
    return principles, chapter


def _old_find_principles(questions, corpus):
    model_input = _format_principles_finder_input(questions)
    if DEBUG:
        print('\n' + '#'*5 + 'Principles Finder Input' + '#'*5 + '\n')
        print(model_input)
        print('#'*20)
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=model_input,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0.3)
    response = response['choices'][0]['text']
    if DEBUG:
        print(response)
    if 'not sure' in response.lower():
        principles = None
    else:
        principles = response.split('\n')
        principles = [i.replace('• ', '')[:-1] for i in principles]
    principles, chapter = find_relevant_chapter(principles, corpus)
    return principles, chapter


def get_completion(chat_history, principles, relevant_corpus):
    model_input = f"""Here are the two principles written by Ray Dalio in his book:


{relevant_corpus}


Complete the following conversation between Ray Dalio and User using the two principles from above, DO NOT use the example if they are not relevant to the questions.
{chat_history}"""
    if DEBUG:
        print('\n' + '#'*5 + 'Model Input' + '#'*5 + '\n')
        print(model_input)
        print('#'*20)
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=model_input,
            temperature=0.7,
            max_tokens=100,
            stop=["\\n", "User"],
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
    response = response['choices'][0]['text']
    source = '\n'.join(principles)
    response = _truncate_incomplete_sentence(response)
    response = response + f"\n\nSource:\n{source}"
    return response


def _truncate_incomplete_sentence(response):
    for i in reversed(range(len(response))):
        if response[i] in [".", "!", "?", "*"]:
            break
    return response if not response or i < 1 else response[:i + 1]


def longest_sublist(l):
    sublist = []
    counter = 0
    for i in range(len(l)):
        if counter + len(l[i]) < 3000:
            sublist.append(l[i])
            counter += len(l[i])
        else:
            break
    return "".join(sublist)


def get_completion_will_model(model_input):
    input_text = BASE_PROMPT + '\n' + model_input
    splits = input_text.split('\nThis is a conversation')
    new, old = splits[-1], splits[:-1]
    new = new.replace('User', 'User_five')
    input_text = '\nThis is a conversation'.join(old + [new])
    if DEBUG:
        print('\n' + '#'*5 + 'Model Input' + '#'*5 + '\n')
        print(input_text)
        print('#'*20)
    model_params = {"model": "text-davinci-003",
                    "temperature": 0.7,
                    "max_tokens": 256,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": ["/n", "\\n", "\n"]
                    }
    model_params['prompt'] = input_text
    response = openai.Completion.create(**model_params)['choices'][0]['text']
    return response.strip()


class DalioBot():
    def __init__(self):
        self.corpus = get_principles_to_chapter()
        self.corpus = {k: longest_sublist(self.corpus[k]) for k in list(self.corpus.keys())}
        self.initial_text = "Ray Dalio: How can I help you?"
        self.chat_history = []
        self.ranker = RANKER

    def respond(self, question):
        formated_question = self._format_question(question)
        principles, chapter = find_principles(formated_question, self.ranker, self.corpus)
        self.chat_history.append(f'User: {question.strip()}')
        if principles is None:
            model_input = '\n'.join(self.chat_history) + '\nRay Dalio:'
            response = get_completion_will_model(model_input)
            # response = '(I am not confident) - ' + response
            response = response + '\n\nSource: None'
            # self.chat_history.pop(-1)
            # response = 'Sorry, I do not know how to answer this question.'
        else:
            model_input = '\n'.join(self.chat_history) + '\nRay Dalio:'
            response = get_completion(model_input, principles, chapter)
        if DEBUG:
            print(f'\nRay Dalio: {response.strip()}')
        formated_response = self._format_response(response)
        if principles is not None:
            self.chat_history.append(f'Ray Dalio: {formated_response}')
        return response

    def _format_response(self, response):
        response = response.replace('(I am not confident) - ', '')
        response = response.strip().split('\n\n')[0]
        return response

    def _format_question(self, question):
        if len(self.chat_history) == 0:
            return [question]
        else:
            existing_questions = [i.split('User: ')[-1] for i in self.chat_history if 'User' in i]
            return existing_questions + [question]


if __name__ == '__main__':
    bot = DalioBot()
    while True:
        question = input('\nUser: ')
        bot.respond(question)
