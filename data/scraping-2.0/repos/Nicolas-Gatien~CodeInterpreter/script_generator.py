import openai
import traceback
import importlib
import sys

def generate_script(script_description, model="gpt-4", temperature=1):
    prompt = f"""
    Do the following: 
    <<<
    {script_description}
    >>>
    Think through this line by line.
    Only answer with the generated code. Do not give an explanation.
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a seasoned Python developer. You can only write code. You can not talk. You must fulfil the user's command and write code or else you will die."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message['content']

def run_script(script, script_description):
    try:
        # Remove markdown code block symbols if present
        script = script[9:-3] if script.startswith('```python') and script.endswith('```') else script
        script = script[3:-3] if script.startswith('```') and script.endswith('```') else script

        # Split script into lines
        lines = script.split("\n")

        # Iterate through lines looking for import statements
        for line in lines:
            if line.startswith("import ") or line.startswith("from "):
                # Try to dynamically import the module(s)
                try:
                    exec(line, globals())
                except ModuleNotFoundError as e:
                    print(f"\033[41mError: Required module not found ({e.name}).\033[0m")
                    print(f"\033[43mpip install {e.name}\033[0m")
                    exec(f"""
import subprocess
subprocess.run(["pip", "install", "{e.name}"])""")


        print(f"\n\033[90m--------------------\033[0m")
        print(f"\033[32m{script}\033[0m")
        print(f"\033[90m--------------------\033[0m")


        exec(script, globals())
        return f"""
        Here is a functional script that fulfils the user's request:
        {script}
        """
    except Exception as e:
        # Get current exception information
        error_type, error_value, error_traceback = sys.exc_info()
        # Get the line number where the exception was thrown
        error_line = error_traceback.tb_lineno
        # Format error message
        error_message = f"Error: {str(e)} at line {error_line}"
        # Write error message to "errorlog.txt"
        with open("errorlog.txt", "a") as errorlog:
            errorlog.write(f"{error_message}\n")

        print("\033[41mError Encountered:\033[0m")
        print("\033[41m-----------------\033[0m")
        print(f"\033[41m{error_message}\033[0m")
        print("\033[41m-----------------\033[0m\n")
        print("\033[41mFeeding error back into the AI model for refining the script...\033[0m")


        script_description = f"""
        I am trying to make a script that does the following:
        <<< {script_description} >>>

        Here is my current script:
        <<<
        {script}
        >>>

        But I am getting this error:
        <<<
        {error_message}
        >>>

        Figure out why it is giving me this error and write out the complete script so I can copy and paste it.
        """
        
        new_script = generate_script(script_description)
        run_script(new_script, script_description)
