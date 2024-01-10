from openai import InvalidRequestError
import pytest
from autobyteus.llm_integrations.openai_integration.openai_chat_api import OpenAIChatApi
from autobyteus.llm_integrations.openai_integration.openai_message_types import UserMessage, SystemMessage, AssistantMessage
from autobyteus.llm_integrations.openai_integration.openai_models import OpenAIModel

@pytest.mark.skip(reason="Integration test calling the real OpenAI API")
def test_process_input_messages_integration():
    """
    Integration test to check if the process_input_messages method interacts correctly with the OpenAI Chat API.
    """
    api = OpenAIChatApi()
    messages = [UserMessage("Hello, OpenAI!")]
    response = api.process_input_messages(messages)
    assert isinstance(response, AssistantMessage)  # Ensure response is an AssistantMessage instance
    assert isinstance(response.content, str)  # The content of the response should be a string

@pytest.mark.skip(reason="Integration test calling the real OpenAI API")
def test_refine_writing_integration():
    """
    Integration test to check if the process_input_messages method interacts correctly with the OpenAI Chat API for refining writing tasks.
    """
    api = OpenAIChatApi()
    
    system_message = SystemMessage("You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. Knowledge cutoff: September 2021. Please feel free to ask me anything.")
    user_message_content = """
    As an expert in refining writing, your task is to improve the given writing situated within the [Writing] section. The content of the writing is situated within the $start$ and $end$ tokens.

    Follow the steps below, each accompanied by a title and a description:
    1. Analyze the Prompt:
       - Dissect the prompt to understand its content and objectives.
    2. Determine the Domain:
       - Identify the domain to which this prompt belongs.
    3. Evaluate and Recommend Linguistic Enhancements:
       - Articulate your thoughts on the prompt's conciseness, clarity, accuracy, effectiveness, sentence structure, consistency, coherence, word order, content structure, usage of words, etc. If you think there are areas that need to be improved, then share your detailed opinions where and why.
    4. Present the Refined Prompt:
       - Apply your improvement suggestions from step 3 and present the refined prompt in a code block.

    [Writing]
    $start$
    As a top Vue3 frontend engineer, your task is to analyze the error and relevant codes, and based on your analysis results either propose a solution or add more debugging information for further analysis.
    ... (rest of the content)
    $end$
    """
    user_message = UserMessage(user_message_content)
    
    messages = [system_message, user_message]
    response = api.process_input_messages(messages)
    assert isinstance(response, AssistantMessage)  # Ensure response is an AssistantMessage instance
    assert isinstance(response.content, str)  # The content of the response should be a string


@pytest.mark.skip(reason="Integration test calling the real OpenAI API")
def test_refine_writing_integration_real():
    """
    Integration test to check if the process_input_messages method interacts correctly with the OpenAI Chat API for refining writing tasks.
    """
    api = OpenAIChatApi()
    
    system_message = SystemMessage("You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. Knowledge cutoff: September 2021. Please feel free to ask me anything.")
    user_message_content = '''
As a senior Python software engineer, approach the requirements outlined between the `$start$` and `$end$` tokens in the `[Requirement]` section using a two-phase methodology:

Phase 1: Analytical Breakdown
- Understand the essence of the provided requirements.
- Provide a methodical, step-by-step breakdown of your approach.
- Detail design decisions, considerations, or architectural changes.

Phase 2: Summarized Implementation
- After the analytical breakdown, provide the updated source code.
- Adhere to Python PEP8 coding best practices, SOLID principles, and utilize suitable design patterns.
- Incorporate type hints and maintain file-level docstrings updated with any code changes.
- Detail file management decisions: whether to create a new folder or use an existing one. Ensure the naming of files and folders aligns with the requirement's features.
  
  For reference, the current project file structure is:
    autobyteus
        ...
        workspaces
            workspace_service.py
        semantic_code
                embedding
                    openai_embedding_creator.py
    tests
        unit_tests
            ...
            semantic_code
                embedding
                    test_openai_embedding_creator.py
        integration_tests
            ...
            semantic_code
                index
                    test_index_service_integration.py
- Prioritize absolute imports over relative ones.
- Present the code modifications using the following format. For instance:
{
   command: command_type (e.g., update_file),
   file_path: path_to_file,
   content: modified_content
}

  Available commands:
  - update_file: Update a file's content.
      - file_path: The target file.
      - content: The new content.
      
  - update_function: Modify a specific function.
      - file_path: The target file.
      - function_name: The function in question.
      - content: The new function content.

By following this two-phase methodology, the output should be a blend of comprehensive reasoning followed by summary of the actual code changes.

[Requirement]
$start$
Previously we have implemented
File: autobyteus/llm_integrations/openai_integration/openai_gpt_integration.py
```
""
openai_gpt_integration.py: Implements the OpenAIGPTIntegration class which extends the BaseLLMIntegration abstract base class.
This class integrates the OpenAI GPT models (gpt3.5-turbo, gpt4) with the agent program. It uses the OpenAI API to process a list of input messages and return the model's responses.
"""

from autobyteus.config import config
from autobyteus.llm_integrations.openai_integration.base_openai_api import BaseOpenAIApi
from autobyteus.llm_integrations.openai_integration.openai_api_factory import ApiType, OpenAIApiFactory
from autobyteus.llm_integrations.base_llm_integration import BaseLLMIntegration
from autobyteus.llm_integrations.openai_integration.openai_models import OpenAIModel

class OpenAIGPTIntegration(BaseLLMIntegration):
    """
    OpenAIGPTIntegration is a concrete class that extends the BaseLLMIntegration class.
    This class is responsible for processing input messages and returning responses from the OpenAI GPT model.
    
    :param api_type: Type of the OpenAI API to use.
    :type api_type: ApiType
    :param model_name: Name of the OpenAI model to be used. If not provided, the default from the respective API class will be used.
    :type model_name: OpenAIModel, optional
    """

    def __init__(self, api_type: ApiType = ApiType.CHAT, model_name: OpenAIModel = None):
        super().__init__()
        if model_name:
            self.openai_api: BaseOpenAIApi = OpenAIApiFactory.create_api(api_type, model_name)
        else:
            self.openai_api: BaseOpenAIApi = OpenAIApiFactory.create_api(api_type)
    
    def process_input_messages(self, input_messages):
        """
        Process a list of input messages and return the LLM's responses.

        :param input_messages: List of input messages to be processed.
        :type input_messages: list
        :return: List of responses from the OpenAI GPT model
        :rtype: list
        """

        return self.openai_api.process_input_messages(input_messages)  # We're now processing one message at a time

```
But the process_input_messages implemention is already changed. You can learn how to 
use the api now from the following test.
File: tests/integration_tests/llm_integrations/openai_integration/test_openai_chat_api_integration.py
```
@pytest.mark.skip(reason="Integration test calling the real OpenAI API")
def test_refine_writing_integration():
    """
    Integration test to check if the process_input_messages method interacts correctly with the OpenAI Chat API for refining writing tasks.
    """
    api = OpenAIChatApi()
    
    system_message = SystemMessage("You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. Knowledge cutoff: September 2021. Please feel free to ask me anything.")
    user_message_content = """
    As an expert in refining writing, your task is to improve the given writing situated within the [Writing] section. The content of the writing is situated within the $start$ and $end$ tokens.

    Follow the steps below, each accompanied by a title and a description:
    1. Analyze the Prompt:
       - Dissect the prompt to understand its content and objectives.
    2. Determine the Domain:
       - Identify the domain to which this prompt belongs.
    3. Evaluate and Recommend Linguistic Enhancements:
       - Articulate your thoughts on the prompt's conciseness, clarity, accuracy, effectiveness, sentence structure, consistency, coherence, word order, content structure, usage of words, etc. If you think there are areas that need to be improved, then share your detailed opinions where and why.
    4. Present the Refined Prompt:
       - Apply your improvement suggestions from step 3 and present the refined prompt in a code block.

    [Writing]
    $start$
    As a top Vue3 frontend engineer, your task is to analyze the error and relevant codes, and based on your analysis results either propose a solution or add more debugging information for further analysis.
    ... (rest of the content)
    $end$
    """
    user_message = UserMessage(user_message_content)
    
    messages = [system_message, user_message]
    response = api.process_input_messages(messages)
    assert isinstance(response, AssistantMessage)  # Ensure response is an AssistantMessage instance
    assert isinstance(response.content, str)  # The content of the response should be a string
```
Now we need to update the implementation of OpenAIGPTIntegration. The input_messages of strings.
By the way, please use system_message = SystemMessage("You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. Knowledge cutoff: September 2021. Please feel free to ask me anything.")
$stop$

   '''
    user_message = UserMessage(user_message_content)
    
    messages = [system_message, user_message]
    response = api.process_input_messages(messages)
    assert isinstance(response, AssistantMessage)  # Ensure response is an AssistantMessage instance
    assert isinstance(response.content, str)  # The content of the response should be a string


