
import inspect
import os
import re
import time

import black
import ipywidgets as widgets
import openai
import pyautogui
import pyperclip
from colored import attr, fg
from IPython import get_ipython
from IPython.display import display
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer


class AiCodeCorrector:
    """This class will help you fix your code errors."""
    
    def __init__(self,openai_key = None):
        """Initialize the class."""
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.revise_code_snippets = {}
        self.revised_code =''
        self.snippets_index = 0
        self.retry = 0
        try:
            self.Err = get_ipython().user_ns.get('Err')
            self.In = get_ipython().user_ns.get('In')
            
        except Exception:
            raise AttributeError("You must run this code in a Jupyter notebook.")
            
    def format_python_code(self,code):
        """Return a formatted version of Python code."""
        try:
            formatted_code = black.format_str(code, mode=black.FileMode())
        except Exception:
            formatted_code = code
        
        return formatted_code

    def syntax_highlighting(self,code):
        """ This function will format python code in the terminal. 
        Args:
            code: a string, the python code to format
        Returns:
            formatted_code: a string, the formatted code
        """
        # Split the code into lines
        lines = code.split('\n')
        # Initialize the list of formatted lines
        formatted_lines = []
        # Loop through each line of code
        for line in lines:
            # Format the line
            formatted_line = highlight(line, PythonLexer(), TerminalFormatter())
            # Add the formatted line to the list
            formatted_lines.append(formatted_line)
        # Join the lines together
        formatted_code = ''.join(formatted_lines)
        
        return formatted_code


    def _show_function_definition(self, function): 
        """Return the source code for a function."""
        
        try:
            source_lines, _ = inspect.getsourcelines(function)
        except TypeError:
            return "This function does not have a definition."
        except OSError:
            return "This function does not have a definition."
        
        # Join the lines together into a single string.
        function_definition = ''.join(source_lines)
        
        # Format the code.
        formatted_definition = self.format_python_code(function_definition)

        return formatted_definition

    def copy_to_clipboard(self):
        # try to copy the last error message in the Err dict to the clipboard
        try:
            if not self.Err:
                self.Err = get_ipython().user_ns.get('Err')
            pyperclip.copy(self.Err[list(self.Err.keys())[-1]])
        except Exception:
            pass


    def _extract_function_name(self,text):
        # Find the first word after 'def' and before the first '(', if any.
        match = re.search(r'(?:def|)(\w+)\(', text)
        if match:
            # The match object always has a 'group' method.
            function_name = match.group(1)
            return function_name
        else:
            # If there is no match, return an empty string.
            return ''


    def _extract_function_name_from_stacktrace(self,text):
        
        # Find the index of " in " and add 4 to get the start index
        start_index = text.find(" in " ) + 4
        
        # Find the index of the next newline character
        end_index = text.find("\n", start_index)
        
        # Extract the text between the two indices, and strip whitespace
        extracted_text = text[start_index:end_index].strip()
        
        # Return the extracted text
        return self._extract_function_name(extracted_text)


    def _split_string_last_cell_in(self, input_string):
        # Find the last instance of "Cell In" in the string
        index = input_string.rfind("Cell In")
        if index != -1:
            # If there is an instance of "Cell In" in the string,
            # then extract the function name from the stack trace
            stack_trace = input_string[index:].strip()
            func_name = self._extract_function_name_from_stacktrace(stack_trace)
            if func_name:
                return func_name
            else:
                return ''
        else:
            # If there is no instance of "Cell In" in the string,
            # then return an empty string
            return ""   


    def _get_error_function(self):
        """Returns the source code of the last error function in the Err dictionary."""
        
        # Get the name of the last function in the Err dictionary.
        last_function_name = self._split_string_last_cell_in(self.Err[list(self.Err.keys())[-1]])
        # Get the global namespace
        globals = get_ipython().user_ns
        # If the function exists in the global namespace and is callable, get its source code.
        if last_function_name in globals and callable(globals[last_function_name]):
            last_function = globals[last_function_name]
            function_definition = inspect.getsource(last_function)
        
            return function_definition
        
        # If the function doesn't exist, print an error message and return an empty string.
        else:
            return ''

    # This code will emulate the user typing the command "esc b" then pressing enter.
    # Next, the code will emulate the user typing the command "command v" which will paste the previously copied text.

    def _simulate_typing(self,index):
        #change type of cell to code
        pyautogui.press('y')
        # Copy the code snippet to the clipboard
        pyperclip.copy(self.revise_code_snippets[index])
        # Focus on the next cell
        pyautogui.press('tab')
        # Presses esc then b, then enter
        pyautogui.hotkey('esc','a')
        pyautogui.press('enter')
        time.sleep(0.7)
        # Presses command then v
        pyautogui.hotkey('command', 'v')
        
    def _suggest_fix(self,index):
        
        pyautogui.hotkey('shift', 'tab')
        pyautogui.hotkey('shift', 'tab')
        pyautogui.hotkey('shift', 'enter')
        pyautogui.press('esc')
        
    def explain_error(self):
        
        
        openai.api_key = self.openai_key
        
        # loop through all errors and get the last one
        try:
            if not self.Err:
                self.retry += 1
                if self.retry >= 2:
                    raise LookupError(' Don´t forget to install and %load_ext jupyter_ai_magics')
                raise NameError
            else:
                err_index = list(self.Err.keys())[-1]
        except NameError:
    
            print(' Everything is alright, no error to fix')
            return 
        except Exception as e:
            print(e)
            return
        else:
            # get the error_code that produced the error
            error_code = self._get_error_function() or self.In[err_index]    
                
            # get the number of characters in the previous error_code
            char_no = 250 + len(self.Err[err_index]) + len(self.In[err_index]) + len(error_code)# 250 is the length of prompt
            # get the number of characters remaining
            rest_char_no = 4000 - char_no
            # create the prompt message
            prompt = f'\"{self.Err[err_index][-rest_char_no:]} \" was produced from following error_code \"{error_code} \". Explain the error and try your best to examine the source code for possible programming error. Rewrite the code, your code should not be the same as the follwing code: {self.revised_code}. Wrap your rewritten code in 5 angle brackets like this <<<<<  >>>>>, 5 ok! ',
            print('Bot is coding! Please wait for a few seconds...')
            response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=.7,
                    max_tokens=4096-char_no,# 4096 is the max number of characters
                    top_p=1.0,
                    frequency_penalty=0.5,
                    presence_penalty=0.8,
                    stop=["###"]
                    )    
            return response['choices'][0]['text'] # type: ignore

    def revise_code(self,explanation):
        """ This code checks to see if the explanation is valid and extracts the revised code from it.
        Args:
            explanation: a string, the explanation text
        
        Returns:
            revised_code: a string, the revised code or an empty string
        """
        if explanation:
            # Split the explanation at the start and end of the revised code
            split_explanation = explanation.split("<<<<<")
            # Ensure that the explanation has the expected format
            if len(split_explanation) >= 2:
                # Extract and return the revised code
                revised_code = split_explanation[1].split(">>>>>")[0]
                return revised_code
            else:
                print("Invalid explanation format.")
        else:
            print("No explanation provided.")
        
        return ""  # Return an empty string if no revised code is available

    def fix_broken_code(self,explanation = True):
        """ This code will get the error produced by the last line of code and then attempt to fix it."""
        if not self.Err:
            self.Err = get_ipython().user_ns.get('Err') # type: ignore
            
            
        # Get the error and return an explanation
        explained_error = self.explain_error()
        
        # If there is an error, try to fix it
        if explained_error:
            
            # Return the revised code
            self.revised_code = self.revise_code(explained_error)
            
            if explanation:
                
                print("Explanation:", explained_error.split('<<<<<')[0])
                
            # If the revised code exists, format it and copy it to the clipboard
            if self.revised_code:
                self.revise_code_snippets[self.snippets_index] = self.revised_code
            
                print(self.syntax_highlighting(self.revised_code))
    
                # Ask the us er if they want to apply the revised code
                _display_button(self._simulate_typing,self._suggest_fix,self.snippets_index)
                #increment dictionary index
                self.snippets_index += 1
                
            else:
                print('No suggestion')
        else:
            pass
    
def _display_button(func_apply,func_try,snippets_index):
    
    # Create a button widget
    apply_button = widgets.Button(description='Apply code')
    try_button = widgets.Button(description='Try another code')

    # Style the button
    apply_button.style.button_color = 'blue'
    
    click = 0
    # Function to be called when the button is clicked
    def on_button_clicked(b):
        nonlocal click
        click += 1
        # Continue with the desired actions
        if click == 1:
            func_apply(snippets_index)
        else:
            pass
        
    def on_try_button_clicked(b):
        # Code for handling the try button click event
        func_try(snippets_index)
        
    # Register the button click event handler
    apply_button.on_click(on_button_clicked)
    try_button.on_click(on_try_button_clicked)
    # Display the button
    display(try_button)
    display(apply_button)
    