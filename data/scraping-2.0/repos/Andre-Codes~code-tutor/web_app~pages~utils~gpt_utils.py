import openai
from openai import OpenAI
import yaml


class ChatEngine:
    """
    A class for interacting with GPT models via the OpenAI API.
    
    Attributes:
        api_key (str): The OpenAI API key, sourced from environment variables.
        model (str): The GPT model name to be used. Defaults to "gpt-3.5-turbo".
        role_context (str): Operational context for the GPT model, e.g., 'general', 'api_explain'.
        temperature (float): Controls randomness in output. Lower is more deterministic.
        
    Class Variables:
        DISPLAY_MAPPING (dict): Mappings for IPython display function names.
        MD_TABLE_STYLE (str): Default style for Markdown tables.
        
    Methods:
        __init__(): Initializes class attributes.
        set_md_table_style(): Sets Markdown table style.
        get_format_styles(): Prints available response formats.
        get_role_contexts(): Prints available role contexts.
        _validate_and_assign_params(): Validates and assigns prompt and format_style.
        _build_prompt(): Constructs the complete prompt for OpenAI API call.
        _text_api_call(): Makes the API call and stores the response.
        _handle_output(): Handles saving and displaying the response.
        get_response(): Main function to get a response from the GPT model.
        _handle_role_instructions(): Constructs role-specific instructions for the prompt.
        show(): Displays the generated content.
    """

    # Class variables
    # DISPLAY_MAPPING = { # mappings for IPython.display function names
    #     'html': HTML,
    #     'markdown': Markdown
    # }

    MD_TABLE_STYLE = "pipes"  # default format for markdown tables

    def __init__(
            self,
            role_context=None,
            system_role=None,
            temperature=1,
            model="gpt-3.5-turbo",
            stream=False,
            api_key=None,  # os.environ['OPENAI_API_KEY']
            config_path=None):
        """
        Initializes the GPTService class with settings to control the prompt and response.

        # Parameters
        ----------
            role_context (str, optional): Operational context for GPT. This directly controls
            what information is sent to the GPT model in addition to the user's prompt.
            Use the `get_role_contexts()` method to view the available roles. Defaults to 'general'.

            comment_level (str, optional): Level of comment verbosity. Defaults to 'normal'.

            explain_level (str, optional): Level of explanation verbosity. Defaults to 'concise'.

            temperature (float, optional): Controls randomness in output. Defaults to 0.

            model (str, optional): The GPT model name to use. Defaults to "gpt-3.5-turbo".
        """

        if config_path:
            with open(config_path, "r") as f:
                self.CONFIG = yaml.safe_load(f)
        else:
            self.CONFIG = {}

        # Set system role
        self.system_role = system_role or "You're a helpful assistant who answers questions."

        # Set up API access
        self.api_key = api_key

        # Turn off/on streaming of response
        self.stream = stream

        # Set the GPT model
        self.model = model

        # Validate and set role_context
        if self.CONFIG:
            available_role_contexts = self.CONFIG.get('role_contexts', {}).keys()
            self.role_context = role_context if role_context in available_role_contexts else 'general'
        else:
            self.role_context = 'general'

        # Validate and set temperature
        self.temperature = temperature

    def set_md_table_style(self, style):
        available_table_styles = self.CONFIG['response_formats']['markdown']['table_styles'].keys()
        if style not in available_table_styles:
            raise ValueError(
                f"Invalid MD_TABLE_STYLE. Available styles: {list(self.CONFIG['table_formatting'].keys())}.")
        self.MD_TABLE_STYLE = self.CONFIG['response_formats']['markdown']['table_styles'][style]

    def get_format_styles(self):
        available_formats = list(self.CONFIG['response_formats'].keys())
        print("Available response formats:", available_formats)

    def get_role_contexts(self):
        available_role_contexts = list(self.CONFIG['role_contexts'].keys())
        return available_role_contexts

    def _validate_and_assign_params(self, prompt, format_style):
        if prompt is None:
            raise ValueError("Prompt can't be None.")
        self.prompt = prompt
        if format_style:
            self.format_style = format_style.lower()

    def _build_prompt(self):
        """
        Construct the final prompt by combining the role/context specific instructions
        (built from `_handle_role_instructions`) with formatting instructions.

        This method first acquires role/context-specific instructions from the `_handle_role_instructions` method.

        Returns:
            str: The fully constructed prompt, ready to for the API call.

        """

        # get the adjusted prompt reconstructed with role instructions
        prompt = self._handle_role_instructions(self.prompt)

        if self.role_context != 'general':
            response_formats = self.CONFIG.get('response_formats', {})
            format_style = response_formats.get(self.format_style, {})
            response_instruct = format_style.get('instruct', '')

            if self.format_style == 'markdown':
                md_table_style = format_style.get('table_styles', {}).get(self.MD_TABLE_STYLE, '')
                response_instruct += md_table_style
            elif self.format_style == 'html':
                use_css = format_style.get('use_css', False)
                if use_css:
                    css = format_style.get('css', '')
                    response_instruct += css
            # construct and save the final prompt to be sent with API call
            self.complete_prompt = f"{response_instruct}{prompt}"
        else:
            # if no role contexts are available or none are selected
            # the complete prompt defaults to only what is passed
            # to the get_response method
            self.complete_prompt = prompt

    def _text_api_call(self, **kwargs):
        if 'streaming' in kwargs:
            self.stream = kwargs['streaming']
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=self.__messages,
                temperature=self.temperature,
                top_p=0.2,
                stream=self.stream
            )
            if response:
                self.response = response
        except openai.APIConnectionError as e:
            raise e
        except openai.RateLimitError as e:
            raise e
        except openai.APIError as e:
            raise e

    def _vision_api_call(self, prompt):
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What python code is necessary to produce this visualized data?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": prompt,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=700,
            )
            if response:
                self.response = response
        except openai.APIConnectionError as e:
            raise e
        except openai.RateLimitError as e:
            raise e
        except openai.APIError as e:
            raise e

    def _image_api_call(self, prompt):
        try:
            client = OpenAI()
            response = client.images.generate(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            self.response = response
        except openai.APIConnectionError as e:
            raise e
        except openai.RateLimitError as e:
            raise e
        except openai.APIError as e:
            raise e

    def _build_messages(self, prompt, **kwargs):
        # Validate that all items in 'prompt' are strings
        if not all(isinstance(item, str) for item in prompt):
            raise ValueError(f"All elements in the list should be strings {prompt}")
        # Initialize system message
        # override system_role if provided
        if 'system_role' in kwargs:
            self.system_role = kwargs['system_role']
        system_msg = [{"role": "system", "content": self.system_role}]
        # Determine user and assistant messages based on the length of the 'prompt'
        if isinstance(prompt, list) and len(prompt) > 1:
            user_assistant_msgs = [
                {
                    "role": "assistant" if i % 2 else "user",
                    "content": prompt[i]
                }
                for i in range(len(prompt))
            ]
        else:
            user_assistant_msgs = [{"role": "user", "content": self.complete_prompt}]
        # Combine system, user, and assistant messages
        self.__messages = system_msg + user_assistant_msgs

    def _handle_role_instructions(self, user_prompt):
        """
        Construct the context for the prompt by adding role/context specific instructions.

        If no instructions are found in config file, only the `system_role` value
        will be supplied, as this is a necessary arg for the API call.

        Returns:
            str: The inputted prompt with role/context instructions..
        """

        default_documentation = self.CONFIG.get('role_contexts', {}).get('defaults', {}).get('documentation', '')

        default_role_instructions = self.CONFIG.get('role_contexts', {}).get('defaults', {}).get('instruct', '')

        default_system_role = self.CONFIG.get('role_contexts', {}).get('defaults', {}).get('system_role', self.system_role)

        documentation = self.CONFIG.get('role_contexts', {}).get(self.role_context, {}).get('documentation', default_documentation)

        role_instructions = self.CONFIG.get('role_contexts', {}).get(self.role_context, {}).get('instruct', default_role_instructions)

        system_role = self.CONFIG.get('role_contexts', {}).get(self.role_context, {}).get('system_role', default_system_role)

        # set the system_role class variable
        self.system_role = system_role
        # construct the prompt by prefixing any role instructions
        # and appending any documentation to the end
        prompt_with_context = f"{role_instructions}{user_prompt}{documentation}"
        #raise AssertionError("prompt with context:" + str(prompt_with_context))
        self.prompt = prompt_with_context

        return prompt_with_context

    def get_response(self, response_type='text',
                     prompt=None, format_style='markdown',
                     raw_output=True, **kwargs):
        """
        Retrieve a response from the OpenAI API in the
         desired format based on the provided parameters.

        Parameters:

        - response_type (str): The type of response to generate; either 'text' or 'image'.
                               This determines whether to use the `ChatCompletion`
                               or the `Image` API.
                               Default is 'text'.
        - prompt (str): The input text to serve as a basis for the OpenAI API response.
        - format_style (str): The format to use for the response, such as 'markdown'.
                              Default is 'markdown'. This will be ignored if the 'Image'
                              API is chosen since `_build_prompt` will not be called.
        - raw_output (bool): Whether to return the raw JSON response or the extracted content
                             found within: `self.response['choices'][0]['message']['content']`
                             Default is True for raw JSON output.
        - **kwargs: Additional keyword arguments for more advanced configurations.
                    e.g. `_build_messages` can accept `system_role`

        Returns:

        - If raw_output is False and response_type is not 'image', returns the 'content' field
          from the 'choices' list in the API response.
        - Otherwise, returns the raw JSON response.
            In the case of the 'image' api an image URL is returned.

        Raises:
        - May raise exceptions based on internal validation methods and API calls.

        Usage Example:

            get_response(response_type='text',
            system_role="You're a comedian",
            prompt='Tell me a joke.',
            raw_output=False)
        """

        # validate and set the instance variable for prompt
        self._validate_and_assign_params(prompt, format_style)
        openai.api_key = self.api_key

        if response_type == 'text':
            self._build_prompt()
            self._build_messages(prompt, **kwargs)
            self._text_api_call(**kwargs)
        elif response_type == 'image':
            prompt = self._handle_role_instructions(prompt)
            self._image_api_call(prompt)
        elif response_type == 'vision':
            self._vision_api_call(prompt)
        # Return finished response from OpenAI
        if not raw_output and not self.stream:
            if response_type == 'text':
                return self.response.choices[0].message.content
            elif response_type == 'image':
                return self.response.data[0].url
            elif response_type == 'vision':
                return self.response.choices[0].message.content

        return self.response
