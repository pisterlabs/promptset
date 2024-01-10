from tasks.abstractTask import AbstractTask
import logging
from utils.assigment_utils import AssigmentUtils 
import time
from openai import OpenAI
from os import environ


class WhoamiTask(AbstractTask):

    def solve_task(self):
        self.assignment_solution = self.process_task_details()
        if self.send_to_aidevs and self.assignment_solution != None:
            logging.info("Sending answer to aidevs... {self.assignment_solution}")
            self.return_answer()
        else:
            logging.info("Won't send to aidevs... END")
    
    def process_task_details(self):
        searched_name = ''
        counter = 0
        hints = []

        while searched_name == '':
            key = AssigmentUtils.get_key_for_assigment(self.task_name)
            print(f"key: {key}")
            hint = self.get_one_hint()
            #logging.info(f"details: {hint}")
            hints.append(hint)
            user_prompt =  {', '.join(hints)}
            logging.info(f"askign ai with prompt: {user_prompt}")
            searched_name = self.get_ai_response(hint)
            counter += 1
            if(self.mock):
                #  sleep just for testing when not asking AI not to get: 
                #  RequestException: An error occurred while making the request: Limit 4 zapytania na 10 sekund per IP
                time.sleep(3)
            if counter > 7:
                logging.info("I'm giving up after 7 tries")
                break
        if searched_name is not  None and searched_name != '':
            return searched_name
        return None
            

    def get_one_hint(self):
        """ fetch one hint from backend

        Returns:
            _type_: str 
        """        
        assignment_body = AssigmentUtils.get_assigment(self.key)
        return assignment_body['hint']

    def get_ai_response(self, user_info: str) -> str:
        """ Get AI response based on hint and return it only if certain it knows answer. 
        Args:
            hint (str): one piece of information about person

        Returns:
            str: person name or '' if not sure
        """        
        if(self.mock):
            return ''
        else: 
            client = OpenAI(api_key=environ.get('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Return person name and surname and nothing else, based on provided by user information. If you are not absolutelly sure about answer, return nothing."},
                    {"role": "user", "content": user_info}
                ],
                temperature=0,
                max_tokens=20,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n"]
            )
            message = response.choices[0].message.content
            logging.info(f"ai response: {message}")
            return message         

            
        


        
    

# requirements: 
# I need to ask backend for details, it'll return some chunk of data 
# based on this data I need to ask AI to fetch person name based on this data
# if data is not sufficient I need to ask backend again for more data → max 5-6 should be enough 
# every 2 seconds I need to refresh token, because otherwise it'll expire and I have to start over 

 
    