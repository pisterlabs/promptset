from OpenAI_API import QA
from DeepL_API import translate
from QA_model_SQUAD2 import QA_SQUAD
from tqdm import tqdm
import pickle
from termcolor import colored
import colorama
from time import time, sleep

colorama.init(autoreset=True)

def QA_model(question_PL, lang='PL', verbose=False, return_articles=False):
    if lang == 'PL':
        question_EN = translate(question_PL, "EN-GB")
    else:
        question_EN = question_PL

    if verbose:
        print(f'EN: {question_EN}')
    prompts = [
        lambda q: f"Q: {q}\nA: ",
        lambda q: f"{q}\nThe answer is "
        # lambda q: f"{q}\nThe answer goes as follows.\n"
    ]
    preview = []
    for prompt in prompts:
        preview.append(QA(question_EN, prompt))


    answers_EN = ''
    if preview[0]:
        answers_EN += f'{question_EN} {preview[0]}\n'
    if preview[1]:
        answers_EN += f'{question_EN} The answer is {preview[1]}\n'
    # if preview[2]:
    #     answers_EN += f'{question_EN} The answer goes as follows {preview[2]}'

    if not answers_EN:
        if return_articles:
            return dict(), ""
        return ""

    if verbose:
        print(f"EN answers: {answers_EN}")

    answer_EN = QA_SQUAD(question_EN, answers_EN)
    if verbose:
        print(f'SQUAD_2 model: {answer_EN}')

    answer_PL = translate(answer_EN['answer'], "PL")

    if return_articles:
        return answers_EN, answer_PL
    return answer_PL


def run_test(lang='PL'):
    print('Ready to go!')
    while True:
        question_PL = input()
        answer_PL = QA_model(question_PL, lang=lang, verbose=True)
        print(answer_PL)


def run_test_from_file(name, lang='PL', verbose=False, save_articles=False):
    questions = [question.rstrip() for question in open(f'./data/{name}', 'r', encoding='UTF-8')]
    answers = []
    articles = dict()

    file = open(f'./predictions/answers_A_prime_EN.txt', 'a', encoding='UTF-8')
    t_0 = time()
    count = 0
    print(12)
    for question in tqdm(questions[669:]):
        # if time() - t_0 > 60:
        #     t_0 = time()
        #     count = 0
        # if count > 28:
        #     sleep(60 - (time() - t_0))
        # count += 1

        if save_articles:
            article, answer = QA_model(question, lang=lang, verbose=verbose, return_articles=save_articles)
            articles[question] = article
        else:
            answer = QA_model(question, lang=lang, verbose=verbose)
        print('\n'*20)
        print(colored(question,'green'))
        print(answer)
        file.write(f'{answer}\n')
        answers.append(' '.join(answer.split('\n')))
        sleep(3)

    file.close()

    if save_articles:
        with open('./predictions/answers_A_prime_EN.pickle', 'wb') as file:
            pickle.dump(articles, file)


# run_test(lang="EN")

# run_test_from_file("questions_sample.txt", verbose=True, save_articles=True)

run_test_from_file("questions_A_prime_EN.txt", lang='EN', save_articles=True)
