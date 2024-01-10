import logging
from openai_client import OpenAIClient


class LeetCodeProcessor:
    def __init__(self, client: OpenAIClient):
        self.client = client

    def pretty_print(self, messages):
        try:
            for message in messages:
                if message.role == "assistant":
                    return message.content[0].text.value
        except Exception as e:
            logging.error(f"Error in pretty_print: {e}")
            raise

    def process_problem(self, problem_number):
        try:
            problem_statement = f"leetcode problem number {problem_number}"
            thread, run = self.client.create_thread_and_run(problem_statement)
            run = self.client.wait_on_run(run, thread)
            response = self.client.get_response(thread)
            output = self.pretty_print(response)
            return problem_number, output
        except Exception as e:
            logging.error(f"Error processing problem number {problem_number}: {e}")
            return problem_number, None
