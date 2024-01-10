import os
import openai
import json
import datetime
import re
from IPython.display import display, HTML, Markdown

# Load instructions from JSON file
with open("instructions.json", "r") as f:
    INSTRUCTIONS = json.load(f)

class CodeTutor:
    """
    A class for interacting with GPT models via the OpenAI API.
    
    Attributes:
        api_key (str): The OpenAI API key, sourced from environment variables.
        model (str): The GPT model name to be used. Defaults to "gpt-3.5-turbo".
        role_context (str): Operational context for the GPT model, e.g., 'basic', 'api_explain'.
        prompt_context (bool): Whether additional context will be provided in the prompt. 
        comment_level (str): Level of comment verbosity.
        explain_level (str): Level of explanation verbosity.
        temperature (float): Controls randomness in output. Lower is more deterministic.
        
    Class Variables:
        DISPLAY_MAPPING (dict): Mappings for IPython.display function names.
        MD_TABLE_STYLE (str): Default style for Markdown tables.
        
    Methods:
        __init__(): Initializes class attributes.
        set_md_table_style(): Sets Markdown table style.
        get_format_styles(): Prints available response formats.
        get_role_contexts(): Prints available role contexts.
        _validate_and_assign_params(): Validates and assigns prompt and format_style.
        _build_prompt(): Constructs the complete prompt for OpenAI API call.
        _make_openai_call(): Makes the API call and stores the response.
        _handle_output(): Handles saving and displaying the response.
        get_response(): Main function to get a response from the GPT model.
        _handle_role_instructions(): Constructs role-specific instructions for the prompt.
        show(): Displays the generated content.
    """
    
    # Class variables
    DISPLAY_MAPPING = { # mappings for IPython.display function names
        'html': HTML,
        'markdown': Markdown
    }
    
    MD_TABLE_STYLE = "pipes" # default format for markdown tables
    
    def __init__(
        self, 
        role_context=None,
        prompt_context=False,
        comment_level=None,
        explain_level=None,
        temperature=0,
        model="gpt-3.5-turbo"):
        """
        Initializes the GPTService class with settings to control the prompt and response.

        # Parameters
        ----------
            role_context (str, optional): Operational context for GPT. This directly control \
                what is sent to the GPT model in addition to the user inputted prompt. \
                    Use the `get_role_contexts()` method to view the available roles. \
                        Defaults to 'basic'.
            prompt_context (bool, optional): Whether additional context will be provided; \
                typically as API documentation or code. Defaults to False.
            comment_level (str, optional): Level of comment verbosity. Defaults to 'normal'.
            explain_level (str, optional): Level of explanation verbosity. Defaults to 'concise'.
            temperature (float, optional): Controls randomness in output. Defaults to 0.
            model (str, optional): The GPT model name to use. Defaults to "gpt-3.5-turbo".
        """
        
        # Set up API access
        self.api_key = os.environ['OPENAI_API_KEY']
        openai.api_key = self.api_key
        
        # Set the GPT model
        self.model = model
        
        # Validate and set role_context
        available_role_contexts = INSTRUCTIONS.get('role_contexts', {}).keys()
        self.role_context = role_context if role_context in available_role_contexts else 'basic'
        
        # Validate and set prompt_context
        if not isinstance(prompt_context, bool):
            raise ValueError("prompt_context must be a boolean value: True or False")
        self.prompt_context = prompt_context
        
        # Validate and set comment_level
        comment_levels = INSTRUCTIONS['comment_levels']
        self.comment_level = comment_level if comment_level in comment_levels \
            or comment_level is None else 'normal'
        
        # Validate and set explain_level
        explain_levels = INSTRUCTIONS['explain_levels']
        self.explain_level = explain_level if explain_level in explain_levels \
            or explain_level is None else 'concise'
        
        # Validate and set temperature
        if 0 <= temperature <= 1:
            self.temperature = temperature
        else:
            raise ValueError("temperature must be between 0 and 1")

    
    def set_md_table_style(self, style):
        available_table_styles = INSTRUCTIONS['response_formats']['markdown']['table_styles'].keys()
        if style not in available_table_styles:
            raise ValueError(f"Invalid MD_TABLE_STYLE. Available styles: {list(INSTRUCTIONS['table_formatting'].keys())}.")
        self.MD_TABLE_STYLE = INSTRUCTIONS['response_formats']['markdown']['table_styles'][style]
        
    def get_format_styles():
        available_formats = list(INSTRUCTIONS['response_formats'].keys())
        print("Available response formats:", available_formats)
         
    def get_role_contexts():
        available_role_contexts = list(INSTRUCTIONS['role_contexts'].keys())
        print("Available role contexts:", available_role_contexts)

    def _validate_and_assign_params(self, prompt, format_style):
        if prompt is None:
            raise ValueError("Prompt can't be None.")
        self.prompt = prompt
        self.format_style = format_style.lower()

    def _build_prompt(self):
        self.system_role, user_content = self._handle_role_instructions(self.prompt)

        response_instruct = INSTRUCTIONS['response_formats'][self.format_style]['instruct']
        if self.format_style == 'markdown':
            response_instruct += INSTRUCTIONS['response_formats']['markdown']['table_styles'][self.MD_TABLE_STYLE]
        elif self.format_style == 'html':
            response_instruct += INSTRUCTIONS['response_formats']['html']['css']

        self.complete_prompt = f"{response_instruct}; {user_content}"

    def _make_openai_call(self):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.__messages,
            temperature=self.temperature,
        )
        self.response_content = response['choices'][0]['message']['content']

    def _handle_output(self, save_output, print_raw, **kwargs):
        only_code = kwargs.get('only_code', False)
        
        file_exts = {
            "markdown": "md",
            "html": "html"
        }
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # for output file name
        response_file = f"{self.role_context}_{timestamp}.{file_exts[self.format_style]}"
        
        if self.response_content:
            if save_output:
                with open(response_file, 'w') as f:
                    f.write(self.response_content)
            if print_raw:
                print(self.response_content)
            self.show(content=self.response_content, only_code=only_code)
        else:
            print("No response content.")
            
    def _build_messages(self, prompt):
        # Validate that all items in 'prompt' are strings
        if not all(isinstance(item, str) for item in prompt):
            raise ValueError("All elements in the list should be strings")
        
        # Initialize system message
        system_msg = [{"role": "system", "content": self.system_role}]
        
        # Determine user and assistant messages based on the length of the 'prompt'
        if len(prompt) > 1:
            user_assistant_msgs = [
                {
                    "role": "assistant" if i % 2 == 0 else "user", 
                    "content": prompt[i]
                }
                for i in range(len(prompt))
            ]
        else:
            user_assistant_msgs = [{"role": "user", "content": self.complete_prompt}]
        
        # Combine system, user, and assistant messages
        self.__messages = system_msg + user_assistant_msgs

    def get_response(
            self,
            prompt=None,
            format_style='markdown',
            save_output=False,
            print_raw=False,
            **kwargs
        ):
        # _build_messages requires prompt to be a list
        # convert prompt to a list if it is not already
        prompt = [prompt] if not isinstance(prompt, list) else prompt
        self._validate_and_assign_params(prompt, format_style)
        self._build_prompt()
        self._build_messages(prompt)
        self._make_openai_call()
        self._handle_output(save_output, print_raw, **kwargs)
    
    def _handle_role_instructions(self, user_prompt):
        if self.role_context != 'basic':
            prompt_context_key = 'prompt_context_true' if self.prompt_context else 'prompt_context_false'
            prompt_context = INSTRUCTIONS['role_contexts'][self.role_context][prompt_context_key]

            comment_level = f"Provide {self.comment_level}" if self.comment_level is not None else "Do not add any"
            explain_level = f"Provide {self.explain_level}" if self.explain_level is not None else "Do not give any"
            default_documentation = (
                f"{comment_level} code comments and {explain_level} explanation of the process."
            )

            documentation = (
                INSTRUCTIONS.get('role_contexts', {})
                            .get(self.role_context, {})
                            .get('documentation', default_documentation)
            )

            instructions = (
                f"{prompt_context} {INSTRUCTIONS['role_contexts'][self.role_context]['instruct']}"
            )
            user_content = f"{instructions}: {user_prompt}; {documentation}"

            system_role = INSTRUCTIONS['role_contexts'][self.role_context]['system_role']
        else:
            system_role = "You're a helpful assistant who answers my questions"
            user_content = user_prompt

        return system_role, user_content
        
    def show(self, content=None, only_code=False):
        if not self.response_content:
            print("No response to show.")
            return
            
        display_class = self.DISPLAY_MAPPING.get(self.format_style, None)
        
        if not content:
            content = self.response_content
        
        if only_code:
            pattern = r'(```.*?```)'
            matches = re.findall(pattern, content, re.DOTALL) 
            content = '\n'.join(matches)
            
        if display_class:
            display(display_class(content))
        else:
            print("Unknown format.")
