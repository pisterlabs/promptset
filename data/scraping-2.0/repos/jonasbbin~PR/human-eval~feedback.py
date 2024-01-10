from typing import Any
import openai
import math
import random
import asyncio

#Class to handle the feedback step using the previous answer in the self-refinement structure
#The Structure of this class is chosen such that it is similiar to the original self-refinement structure
class HumanEvalFeedback():
    def __init__(self, model, training_prompt) -> None:
        self.model = model
        self.training_prompt = training_prompt

    async def __call__(self, problem, history, temperature=1, top_p=1) -> Any:
        """
        Handle the initial request
        problem string: The initial problem from a dataset
        history string: The conversation of refinement steps
        """
        messages = [
        {"role": "system", "content": "Find the error in the last response of the following problem. Give Feedback to the proposed solution. If there is no error write 'solution is correct'"},
        {"role": "user", "content": history}
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
                                    temperature = temperature
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
        answer = c["choices"][0]["message"]["content"]#.split()#This just extracts the answer
        
        solved = False
        if "it is correct" in answer.lower() or "is no error in the code" in answer.lower():
            solved = True
        if "solutions are correct" in answer.lower() or "solution is correct" in answer.lower() or "response is correct" in answer.lower():
            solved = True

        return answer, history +"\n Feedback: " + answer, solved #Answer as well as history
