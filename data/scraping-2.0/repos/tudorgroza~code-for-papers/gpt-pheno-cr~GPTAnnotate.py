from os.path import join

import openai

from Constants import OPENAI_ORG, OPENAI_KEY
from Prompts import PROMPT_7_TASK, PROMPT_7_INTRO, PROMPT_ALIGN


class GPTAnnotate:
    outputFolder = None
    model = None

    def __init__(self, outputFolder, model):
        self.outputFolder = outputFolder
        self.model = model

    def annotate(self, content, file, prompt, withTripleBackticks=True, writeToDisk=True):
        openai.organization = OPENAI_ORG
        openai.api_key = OPENAI_KEY
        res = None
        callContent = '```' + content + '```'
        if not withTripleBackticks:
            callContent = content

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {'role': 'system',
                     'content': prompt},
                    {'role': 'user',
                     'content': callContent}
                ],
                temperature=0
            )
            res = response.choices[0].message.content
        except RuntimeError as re:
            print(' - Error when processing [{}]: {}'.format(file, re))
        if res:
            if writeToDisk:
                with open(join(self.outputFolder, file), 'w') as fh:
                    fh.write(res)
        return res

    def run(self, textCorpus, prompt):
        count = 1
        for file in textCorpus:
            print(' - {} of {}: {}'.format(count, len(textCorpus), file))
            self.annotate(textCorpus[file], file, prompt)
            count += 1

    def runNERPlus(self, textCorpus, prompt):
        count = 1
        for file in textCorpus:
            print(' - {} of {}: {}'.format(count, len(textCorpus), file))
            result = self.annotate(textCorpus[file], file, prompt, writeToDisk=False)
            if result:
                self.annotate(result, file, PROMPT_ALIGN)
            count += 1

    def runWithBackground(self, textCorpus, backgroundFile):
        with open(backgroundFile, 'r') as fh:
            backgroundContent = fh.read().strip()

        prompt = PROMPT_7_INTRO + '\n' + backgroundContent + '\n\n' + PROMPT_7_TASK

        count = 1
        for file in textCorpus:
            print(' - {} of {}: {}'.format(count, len(textCorpus), file))
            self.annotate(textCorpus[file], file, prompt, withTripleBackticks=False)
            count += 1
