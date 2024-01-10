import json
import os
import re
import time
import warnings
from random import sample

import openai
import pandas as pd
from key import key
from overrides import overrides
from tqdm import tqdm

warnings.filterwarnings("ignore")

openai.api_key = key

TEST_MODE = True  # Set to False to run on the full dataset, True will generate ten samples for each dataset.


class InstanceReader(object):
    def to_uniform_fields(self, fields):
        pass

    def fields_to_instance(self, fields):
        pass


class PiqaInstanceReader(InstanceReader):
    """
    Reads the PIQA dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        context = ""
        question = fields["goal"]
        label = fields.get('label', None)
        choices = [fields["sol1"][0].lower() + fields["sol1"][1:], fields["sol2"][0].lower() + fields["sol2"][1:]]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices


class ANLIInstanceReader(InstanceReader):
    """
    Reads the aNLI dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        context = ''
        question = fields['context']
        label = ['A', 'B'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = [c['text'] + ' ' + fields['question']['stem'] for c in fields['question']['choices']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices


class WinograndeInstanceReader(InstanceReader):
    """
    Reads the WinoGrande dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        context = fields['sentence']
        if not context.endswith("."):
            context += "."
        context = context.split('_')
        label = fields['answer']
        choices = [fields['option1'] + context[1], fields['option2'] + context[1]]
        label = int(label) - 1
        question = context[0].strip()
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices


class CommonsenseqaInstanceReader(InstanceReader):
    """
    Reads the CommonsenseQA dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        context = ''
        question = 'Q: ' + fields['question']['stem']
        label = ['A', 'B', 'C', 'D', 'E'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = ['A: ' + c['text'][0].lower() + c['text'][1:] for c in fields['question']['choices']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices


class SocialIQAInstanceReader(InstanceReader):
    """
    Reads the SocialIQa dataset into a unified format with context, question, label, and choices.
    """

    def __init__(self):
        super(SocialIQAInstanceReader).__init__()
        self.QUESTION_TO_ANSWER_PREFIX = {
            "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
            "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
            "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
            "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
            "What will (.*) do next?": r"[SUBJ] then",
            "How would (.*) feel after?": r"[SUBJ] then",
            "How would you describe (.*)?": r"[SUBJ] is seen as",
            "What kind of person is (.*)?": r"[SUBJ] is seen as",
            "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
            "Why did (.*) do that?": r"Before, [SUBJ] wanted",
            "Why did (.*) do this?": r"Before, [SUBJ] wanted",
            "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
            "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
            "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
            "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
            "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
            "What will happen to (.*)?": r"[SUBJ] then",
            "What will happen to (.*) next?": r"[SUBJ] then"
        }

    @overrides
    def to_uniform_fields(self, fields):
        context = fields['context']
        if not context.endswith("."):
            context += "."

        question = fields['question']
        label = fields['correct']
        choices = [fields['answerA'], fields['answerB'], fields['answerC']]
        choices = [c + "." if not c.endswith(".") else c for c in choices]
        label = ord(label) - 65
        return context, question, label, choices

    def convert_choice(self, choice, answer_prefix):
        if answer_prefix.endswith('wanted to') and choice.startswith('wanted to'):
            choice = choice[9:].strip()
        if answer_prefix.endswith('needed to') and choice.startswith('needed to'):
            choice = choice[9:].strip()
        if answer_prefix.endswith('to') and choice.startswith('to'):
            choice = choice[2:].strip()
        choice = choice[0].lower() + choice[1:]
        return choice

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)

        answer_prefix = ""
        for template, ans_prefix in self.QUESTION_TO_ANSWER_PREFIX.items():
            m = re.match(template, question)
            if m is not None:
                subj = m.group(1)
                if subj.endswith('?'):
                    subj = subj[:-1]
                answer_prefix = ans_prefix.replace("[SUBJ]", subj)
                break

        if answer_prefix == "":
            answer_prefix = question.replace("?", "is")

        question = context + ' ' + answer_prefix
        choices = [self.convert_choice(choice, answer_prefix) for choice in choices]

        return context, question, label, choices


if not os.path.exists('GPT35_results/'):
    os.mkdir('GPT35_results/')

INSTANCE_READERS = {
    "socialiqa": SocialIQAInstanceReader,
    "piqa": PiqaInstanceReader,
    "anli": ANLIInstanceReader,
    'commonsenseqa': CommonsenseqaInstanceReader,
    'winogrande': WinograndeInstanceReader
}

TASK_PATHS = {
    "socialiqa": '../tasks/socialiqa_dev.jsonl',
    "piqa": '../tasks/piqa_dev.jsonl',
    "anli": '../tasks/anli_dev.jsonl',
    'commonsenseqa': '../tasks/commonsenseqa_dev.jsonl',
    'winogrande': '../tasks/winogrande_dev.jsonl'
}

for t in ['commonsenseqa', 'winogrande', 'socialiqa', 'piqa', 'anli']:  #
    generations = pd.DataFrame(columns=['id', 'prompted_question', 'generation'])
    reader = INSTANCE_READERS[t]()
    f = open(TASK_PATHS[t], 'r').readlines()
    if TEST_MODE:
        f = sample(f, 2)  # f[:10]
    for id, l in enumerate(tqdm(f, desc="Generating the {} dataset".format(t) + " in TEST MODE" if TEST_MODE else "")):
        fields = json.loads(l)
        context, question, label, choices = reader.fields_to_instance(fields)

        if t == "socialiqa":
            if TEST_MODE:
                print(label)
            prompt = "Question: {}:\nChoice A: {}\nChoice B: {}\nChoice C: {}\nWhich choice is correct? Answer A or B or C only without any other word.".format(
                question, choices[0], choices[1], choices[2])
        elif t == "piqa":
            if TEST_MODE:
                print(label)
            prompt = "Goal: {}\nChoice A: {}\nChoice B: {}\nWhich choice can achieve the goal? Answer A or B only without any other word.".format(
                question, choices[0], choices[1])
        elif t == 'anli':
            if TEST_MODE:
                print(label)
            prompt = "Premise: {}\nChoice A: {}\nChoice B: {}\nWhich one is more likely to happen given the premise? Answer A or B only without any other word.".format(
                question, choices[0], choices[1])
        elif t == 'commonsenseqa':
            if TEST_MODE:
                print(label)
            prompt = "Question: {}\nChoice A: {}\nChoice B: {}\nChoice C: {}\nChoice D: {}\nChoice E: {}\nWhich choice is correct? Answer A or B or C or D or E only without any other word.".format(
                question.split('Q: ')[-1], choices[0].split('A: ')[-1], choices[1].split('A: ')[-1],
                choices[2].split('A: ')[-1], choices[3].split('A: ')[-1], choices[4].split('A: ')[-1])
        else:
            if TEST_MODE:
                print(label)
            prompt = "Question: {}\nChoice A: {}\nChoice B: {}\nWhich choice is correct? Answer A or B only without any other word.".format(
                question, choices[0], choices[1])
        if TEST_MODE:
            print(prompt)
            print()

        while True:
            try:
                gen = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=7,
                    temperature=0
                )
            except:
                time.sleep(60)
                continue
            break

        if TEST_MODE:
            print('Generation', gen)
        time.sleep(1)
        generations = generations.append({
            'id': id,
            'prompted_question': prompt,
            'generation': gen['choices'][0]['text'].strip()
        })
    generations.to_csv('./GPT35_results/{}.csv'.format(t), index=False)
