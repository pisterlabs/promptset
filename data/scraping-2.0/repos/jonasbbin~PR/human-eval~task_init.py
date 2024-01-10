from typing import Any
import openai
import math
import random
import asyncio
import async_timeout
#Class to handle the intial in the self-refinement structure
#The Structure of this class is chosen such that it is similiar to the original self-refinement structure
class HumanEvalTaskInit():
    def __init__(self, model, training_prompt):
        self.model = model
        self.training_prompt = training_prompt

    async def __call__(self, problem, instruction="blabla", temperature=1, top_p=1) -> Any:
        """
        Handle the initial request
        problem string: The initial problem from a dataset
        """
        test_question = "\n" + problem + "\n"
        prompt = self.training_prompt + "" + test_question #CHANGE BACK TO INPUT
        if instruction == "blabla":
            instruction = "You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
        else:
            pass
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}]

        retries = 0
        max_retries = 2
        error_occured = True #Makes it easier to handle, gets set to False before first call

        while (error_occured and retries<=max_retries):
            error_occured = False

            try:
                async with async_timeout.timeout(60):
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
            return ""
        answer = c["choices"][0]["message"]["content"]#.split()This just extracts the answer
        
        return answer,  test_question + "\n" + "Answer attempt:"+answer
    
    
