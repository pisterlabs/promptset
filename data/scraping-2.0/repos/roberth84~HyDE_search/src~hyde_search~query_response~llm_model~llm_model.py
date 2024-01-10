import os
import json
import openai


class LLM_Model():
    
    def __init__(self):
        """
        LLM_Model is used to query the large language model (LLM)
        """
        openai.api_key_path = '/home/robert_harrington84/HyDE_search/openai_key.txt'

    def get_response_for_direct_question(self, question: str):
        """
        Provides response for questions with hard-coded prompt.

        :param: question: The question to be answered.
        :return: The answer from the LLM for the question.
        """

        prompt = f"""I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: {question}\nA:"""
        response_text = ''
        while response_text == '':
            response_text = self.get_response(prompt)
        return response_text

    def get_response(self, prompt: str):
        """
        Gets response from LLM for a given prompt.

        :param: prompt: The prompt to pass directly to the LLM.
        :return: The answer from the LLM for the prompt.
        """

        response = openai.Completion.create(
              model="text-davinci-003",
              prompt=prompt,
              temperature=0.7,
              max_tokens=200,
              top_p=1,
              frequency_penalty=0.0,
              presence_penalty=0.0
            )
        return response['choices'][0]['text'].strip()