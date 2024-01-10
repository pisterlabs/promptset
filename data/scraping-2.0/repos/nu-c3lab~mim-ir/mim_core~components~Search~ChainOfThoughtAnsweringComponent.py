'''
This file is part of Mim.
Mim is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
Mim is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Mim.
If not, see <https://www.gnu.org/licenses/>.
'''
"""
Chain of Thought Answering Component
"""

import re
import pandas as pd
from openai.error import RateLimitError

from typing import Dict
from mim_core.structs.Step import Step
from mim_core.components.Search.HighLevelQAComponent import HighLevelQAComponent
from mim_core.components.Search.GPT35Interface import GPT35Interface

from mim_core.exceptions import UnhandledOperationTypeError

class ChainOfThoughtAnsweringComponent(HighLevelQAComponent):
    """
    This search component uses a GPT-3.5 with chain of thought reasoning
    to answer a given question.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = GPT35Interface()

    def answer_with_chain_of_thought(self,
                                     question: str) -> str:
        question_answered = False
        while not question_answered:
            try:
                prompt = "Example" \
                         + "\nQuestion: Which country contains the river that is longest among those in the continent having the tallest mountain?" \
                         + "\nGive the answer as a single noun phrase. Let's think step by step." \
                         + "\n\nFirst, we need to identify the continent with the tallest mountain. The tallest mountain in the world is Mount Everest, which is located in the continent of Asia." \
                         + "\n\nNext, we need to determine which country contains the longest river in Asia. The longest river in Asia is the Yangtze River, which is located in China." \
                         + "\n\nAnswer: China" \
                         + "\n\nUsing the form of the example above, answer the following question." \
                         + f"\nQuestion: {question}" \
                         + "\nGive the answer as a single noun phrase. Let's think step by step."

                result = self.model.generate(prompt)
                match = re.search(r'Answer: (.*)$', result)
                answer = match.group(1) if match else 'no'
                question_answered = True
            except RateLimitError as e:
                print('Hit rate limit. Retrying')
            except:
                answer = 'error'
                question_answered = True
        return answer

    def answer(self,
               step: Step,
               timing: Dict = None) -> pd.DataFrame:
        """
        A function that wraps access to the core searching operations: select, project, filter.
        :param step: The step for which to carry out the select operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return: Pandas Dataframe containing the answers.
        """

        # Initialize timing fields
        if timing:
            if "answer_engine" not in timing:
                timing["answer_engine"] = {}
            if str(self.__class__.__name__) not in timing["answer_engine"]:
                timing["answer_engine"][str(self.__class__.__name__)] = {"total": 0}
            if "document_retrieval" not in timing["answer_engine"][str(self.__class__.__name__)]:
                timing["answer_engine"][str(self.__class__.__name__)]["document_retrieval"] = 0
            if "answer_retrieval" not in timing["answer_engine"][str(self.__class__.__name__)]:
                timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] = 0
            if "answer_processing" not in timing["answer_engine"][str(self.__class__.__name__)]:
                timing["answer_engine"][str(self.__class__.__name__)]["answer_processing"] = 0

        try:
            return self.operations[step.operator_type](step, timing)
        except Exception as e:
            raise UnhandledOperationTypeError(step.operator_type)
