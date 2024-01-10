from typing import Any
import openai
import math
import random
import asyncio
#ToDo:
#Early Stopping
#History and prompt curently double

#Class to handle the refinement step using the feedback in the self-refinement structure
#The Structure of this class is chosen such that it is similiar to the original self-refinement structure
class HumanEvalTaskIterate():
    def __init__(self, model, training_prompt) -> None:
        self.model = model
        self.training_prompt = training_prompt

    async def check_correct(problem, solution):
        """
        Checks if the solution is already correct and no refining is necessary.
        """
        return False

    async def __call__(self, problem, history, temperature = 1, top_p=1) -> Any:
        """
        Handle the initial request
        problem string: The initial problem from a dataset
        history string: The conversation of refinement steps
        Follow the given examples and answer the question. Use the provided Feedback on the previous response to refine the answer for the same question. Make sure to write your final solution in Python Code
        """
        test_question = "\n " + problem + "\n" + "Python Code:  "
        prompt = self.training_prompt + "\n" + history + test_question
        instruction = "Use the provided Feedback on the previous response to improve the answer for the same question. Just write the Python code"
        messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt + " \n" + "Let's think step by step"}
        ]
        retries = 0
        max_retries = 2
        error_occured = True #Makes it easier to handle, gets set to False before first call

        while (error_occured and retries<=max_retries):
            error_occured = False

            try:
                c = openai.ChatCompletion.acreate(
                                    model=self.model,
                                    messages=messages,
                                    top_p = top_p,
                                    temperature=temperature
                                )
                c = await c #Waits on the response
            except openai.error.RateLimitError as e:
                print(f"Request exceeded rate limit: Error Code {e.code}")
                retries += 1
                error_occured = True
                raise e
            except openai.error.APIError as e:
                print(f"Another API error: {type(e)} with code {e.code}")
                error_occured = True
                retries += 1
                wait = random.randint(1,8)
                await asyncio.sleep(wait)
            except openai.error.ServiceUnavailableError as e:
                print(f"Server Unavailable. Error Code {e.code}")
                retries += 1
                error_occured = True
                wait = random.randint(1,8)
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"Got some error when Calling the engine {e}")
                retries+= 1
                error_occured = True

        if (error_occured): #Got no answer
            return None
        answer = c["choices"][0]["message"]["content"]#This just extracts the answer

        return answer, history + "\n" + "Next answer attempt: "+ "\n" + answer #Return answer and updated history
