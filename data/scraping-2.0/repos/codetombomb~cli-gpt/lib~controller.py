from decouple import config
from progress.spinner import PixelSpinner
from prettycli import bright_green, red
import openai
import requests
import asyncio
import concurrent.futures

class Controller:
    def __init__(self):
        self.key = config("OPENAI_API_KEY")
        self.organization = "org-lxPTLC4axXuep5yp0iaq3KB8"
        self.awaiting_response = False

    def start(self):
        openai.organization = self.organization
        openai.api_key = self.key
        
    def post_request(self, question):
        try:
            s = requests.Session()
            s.headers = {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            }
            resp = s.post(
                url="https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant named PyCli-GPT."},
                        {"role": "user", "content": question},
                    ],
                },
                timeout=None
            )
            return {"data": resp.json()["choices"][0]["message"]["content"]}
        except:
            return {"data": str(red("Something went wrong"))}
    
    
    async def print_spinner(self):
        spinner = PixelSpinner(str(bright_green("Generating response ")))
        while self.awaiting_response:
            await asyncio.sleep(.2)
            spinner.next()
        spinner.finish()
        
    async def ask_question(self, question):
        self.awaiting_response = True
        
        async def run_request_in_executor():
            loop = asyncio.get_event_loop()
            
            # https://docs.python.org/3/library/concurrent.futures.html
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
                return await loop.run_in_executor(executor, self.post_request, question)

        resp_future = asyncio.create_task(run_request_in_executor())
        spinner_task = asyncio.create_task(self.print_spinner())
    
        # https://superfastpython.com/asyncio-wait/
        done, _ = await asyncio.wait([resp_future, spinner_task], return_when=asyncio.FIRST_COMPLETED)
    
        for task in done:
            if task != resp_future:
                task.cancel()
    
        data = await resp_future
    
        self.awaiting_response = False
        return data