import json
import re
from concurrent.futures import ThreadPoolExecutor

import openai

from steps.step import AbstractStep


class CheckGrammarStep(AbstractStep):
    def __init__(self, text_file, report_file):
        self._text_file = text_file
        self._report_file = report_file

    def process(self):
        with open(self._text_file.full_path(), 'r') as f:
            text = f.read()

        sentences = re.split(r'(?<=[.!?])\s+', text)

        futures = []
        with ThreadPoolExecutor() as executor:
            for sentence in sentences:
                if not sentence:
                    continue

                future = executor.submit(
                    openai.Completion.create,
                    model='text-davinci-003',
                    prompt=f'Check the sentence and correct mistakes. Give the explanation. "{sentence}". If no '
                           f'correction is needed, return "Correct."',
                    max_tokens=512
                )
                futures.append((sentence, future))

        report = []
        for sentence, future in futures:
            edit = future.result()['choices'][0]['text'].strip()
            if 'Correct.' in edit:
                edit = ""

            report.append({
                'sentence': sentence,
                'edit': edit
            })

        with open(self._report_file.full_path(), 'w') as f:
            json.dump(report, f)
