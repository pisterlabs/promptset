
from typing import List, Optional

from ..pipeconfig import PipeConfig
import openai
import re




class GenModel:
    def __init__(self, pipe_config: PipeConfig) -> None:
        openai.api_key = pipe_config.openai_api_key
        self.pipe_config = pipe_config


    def gen_response(self, query_prompt: str, doc_set: Optional[List[int]] = None) -> List[int]:
        """
        uses openai model to return a ranking of ids
        """
        if self.pipe_config.gen_model_checkpoint == 'text-davinci-003':
            response = openai.Completion.create(
                model=self.pipe_config.gen_model_checkpoint,
                prompt=query_prompt,
                temperature=0,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        else:
            assert doc_set is not None, "doc_set must be provided for gpt-3.5-turbo"
            
            # for gpt-3.5-turbo
            response = openai.ChatCompletion.create(
                model=self.pipe_config.gen_model_checkpoint,
                messages = [{'role': 'user', 'content' : query_prompt}],
                temperature=0.4,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0.2,
                presence_penalty=0.0
            )
            

        if self.pipe_config.gen_model_checkpoint == 'text-davinci-003':
            return self.post_process_chatgpt_response(response)
        return self.post_process_gptturbo_response(response, doc_set=doc_set)


    def post_process_chatgpt_response(self, response):
        """
        could be:
        NCTID 6, NCTID 7, NCTID 5
        NCTID: 6, 7, 5
        6, 7, 5
        '1. 195155\n2. 186848\n3. 194407'
        """
        response_pattern = r"(?:NCTID\:?\s*)? ?(\d+)(?!\.)"
        text = response['choices'][0]['text']
        return [int(s) for s in re.findall(response_pattern, text)]

    def post_process_gptturbo_response(self, response, doc_set: List[int]):
        """
        could be:
        'The most relevant clinical trial for this patient is ID 2, followed by ID 3. The remaining trials are not relevant for this patient's condition.'
        """
        text = response['choices'][0]['message']['content']
        ranking = []
        for substr in text.split():
            if substr.isdigit():
                ranking.append(int(substr))

        # the rest are arbitrarily ranked 
        for ncid in doc_set:
            if ncid not in ranking:
                ranking.append(ncid)
        return ranking




	