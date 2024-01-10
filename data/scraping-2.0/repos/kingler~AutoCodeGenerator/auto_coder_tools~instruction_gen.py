import openai
import time
import termcolor
import tenacity

# define your openai api key
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "sk-..."

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
def generate_instructions(user_input=None, model=None):
    if user_input is None:
        user_input = ''
        print(termcolor.colored("\nThe following a multiline input, you can put as much detail as you want in here.", "green"))
        print(termcolor.colored("\nWhat can I code for you in Python? (Type 'done' in a new line and press 'enter' when you are done)", "yellow"))
        while True:
            line = input()
            if line.strip().lower() == 'done':
                break
            user_input += line + '\n'
    
    if user_input:

        response = openai.ChatCompletion.create(
            model=model,
            stream=True,
            # temperature=0.5,
            top_p=0.2,
            messages=[
                {"role": "system", "content": "You are a detail oriented instruction generator."},
                {"role": "user", "content": f"""
                    Generate detailed, precise instructions defining a python pseudo code for user input.
                    consider what will be neeeded to achieve a perfect python program and detail them in the instructions.
                    consider readibility and simplicity of the code.
                    consider type checking and error handling.
                    don't make the instructions too complicated. 

                    Example user input and generated instructions for pseudo code:

                    EXAMPLE 1:

                    user input:
                    an app which generated python using gpt-4 and runs it to make sure it works iteratively

                generated instructions:
                    
                - make a call to gpt-4 api to get some code based on user instructions
                - parse the python markdown code(```python  .... ```) out of the response and save it to response2.py
                - run this code using subprocess
                - retrieve output and error from the subprocess run
                - feed the output and error along with the original code back to a call to gpt-4 and get a new response
                - this should continue untill no errors are raising
                    
                    EXAMPLE 2:

                    user input:

                    a pygame app which generates a maze and make a circle object find the exit

                generated instructions: 
                    
                - create a pygame with a 800 by 800 window
                - create a maze, make sure the maze has a path to the exit
                - create a circle which automatically moves around the maze and tries to find the exit
                    
                EXAMPLE 3:
                    
                user input:
                    
                a tkinter ui for a database app for a university
                    
                generated instructions:
                    
                - create a python database program for a university
                - use csv files to store and keep track of information
                - create a tkinter UI from which admins can enter necessary information to be stored in the dbs
                - UI should also be able to querry the dbs
                
                REAL user_input FOR THIS PROGRAM:
                user input: 

                {user_input}
                
                now you can respond."""}
            ],
            
        )


        responses = ''

        # Process each chunk
        for chunk in response:
            if "role" in chunk["choices"][0]["delta"]:
                continue
            elif "content" in chunk["choices"][0]["delta"]:
                r_text = chunk["choices"][0]["delta"]["content"]
                responses += r_text
                print(r_text, end='', flush=True)
        
        # save the instructions to user_instructions.txt
        with open('user_instructions.txt', 'w', encoding="utf-8", errors="ignore") as file:
            file.write(responses)

        return responses

# generate_instructions(model="gpt-4")
if __name__ == "__main__":
    generate_instructions(model="gpt-4-0613")