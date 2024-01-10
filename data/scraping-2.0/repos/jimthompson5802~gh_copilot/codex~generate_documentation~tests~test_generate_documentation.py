import pytest
from unittest.mock import call, patch, Mock, mock_open as mock_builtin_open

import openai
from openai import OpenAIError

from generate_documentation import generate_documentation, main


def mock_openai_call(*args, **kwargs):
    """
    Mock a call to the OpenAI API for generating documentation content.

    This function simulates a call to the OpenAI API for generating documentation content
    by returning a mocked response object. It is intended to be used as a replacement for
    actual API calls during testing, allowing controlled testing of code that interacts
    with the OpenAI API.

    Args:
        *args: Positional arguments passed to the function (not used in the mock).
        **kwargs: Keyword arguments passed to the function (not used in the mock).

    Returns:
        A mocked response object that simulates the structure of an OpenAI API response.
        The response includes a mocked message content that represents the generated documentation.

    Mocked Response Structure:
        - MockedResponse (class):
            - choices (list): A list containing a single mocked choice.
                - MockedChoice (class):
                    - message (MockedMessage): A mocked message object representing the generated content.
                        - content (str): The content of the mocked documentation.

    Usage:
        Replace calls to the actual OpenAI API with mock_openai_call() in test scenarios to
        isolate and test the behavior of code that interacts with the OpenAI API.

    """

    # Mocked OpenAI API response
    class MockedChoice:
        class MockedMessage:
            content = 'Mocked documentation content.'
        message = MockedMessage()

    class MockedResponse: 
        choices = [MockedChoice()]

    return MockedResponse()

# This decorator will mock the call to openai.ChatCompletion.create
@patch('openai.ChatCompletion.create', side_effect=mock_openai_call)
def test_generate_documentation(mock_openai):
    """
    Test the generate_documentation function using a mocked OpenAI API response.

    This test function uses the @patch decorator to replace the call to openai.ChatCompletion.create
    with the mock_openai_call function. It then tests the generate_documentation function to ensure
    it correctly interacts with the OpenAI API and generates the expected documentation content.

    Test Steps:
    1. Replace the OpenAI API call with a mock response using the @patch decorator.
    2. Define a prompt and source code for documentation generation.
    3. Expected documentation content is provided by the mocked response.
    4. Call generate_documentation with the prompt and source code.
    5. Assert that the actual generated documentation matches the expected content.

    """

    prompt = "Generate Python documentation for a function"
    source_code = "def greet(name):\n    '''This function greets the given name.'''\n    print('Hello,', name)"
    
    expected_documentation = 'Mocked documentation content.'
    actual_documentation = generate_documentation(prompt, source_code)

    assert actual_documentation == expected_documentation


def mock_openai_failure(*args, **kwargs):
    """
    Mock a failed call to the OpenAI API.

    This function simulates a failed call to the OpenAI API by raising an OpenAIError
    with a simulated error message. It is intended to be used as a replacement for
    actual API calls during testing to simulate error scenarios.

    Args:
        *args: Positional arguments passed to the function (not used in the mock).
        **kwargs: Keyword arguments passed to the function (not used in the mock).

    Raises:
        OpenAIError: Simulated exception indicating a failure in the OpenAI API call.

    Usage:
        Replace calls to the actual OpenAI API with mock_openai_failure() in test scenarios
        to test how your code handles failed API calls and error scenarios.

    """

    raise OpenAIError("Simulated OpenAI API failure")

@patch('openai.ChatCompletion.create', side_effect=mock_openai_failure)
def test_generate_documentation_failure(mock_openai):
    """
    Test the generate_documentation function handling of a failed OpenAI API call.

    This test function uses the @patch decorator to replace the call to openai.ChatCompletion.create
    with the mock_openai_failure function, which simulates a failed OpenAI API call. It then tests
    the generate_documentation function to ensure it handles the failure correctly.

    Test Steps:
    1. Replace the OpenAI API call with a mock response indicating failure using the @patch decorator.
    2. Define a prompt and source code for documentation generation.
    3. Use pytest's raises context manager to expect an OpenAIError with a specific message.
    4. Call generate_documentation with the prompt and source code, which should raise the expected exception.

    """
    
    prompt = "Generate Python documentation for a function"
    source_code = "def greet(name):\n    '''This function greets the given name.'''\n    print('Hello,', name)"

    # pytest's raises checks that the expected exception is raised
    with pytest.raises(OpenAIError, match="Simulated OpenAI API failure"):
        generate_documentation(prompt, source_code)


# Another variant for unit tests
# using this prompt: using pytest generate a unit test for the function generate_documentation
def mock_openai_response():
    """
    Generate a mock response object for a successful OpenAI API call.

    This function creates a mock response object that simulates a successful
    call to the OpenAI API for generating documentation content. The response
    includes a mocked message with generated documentation content.

    Returns:
        MockResponse: A mocked response object that simulates a successful API call.
            - choices (list): A list containing a single mocked message with content.
                - Mock (class): A mocked message object.
                    - message (Mock): A mocked message object.
                        - content (str): The content of the generated documentation.

    Usage:
        Use this mock function in test scenarios to simulate a successful OpenAI API response
        and test the behavior of code that interacts with the OpenAI API.

    """

    class MockResponse:
        def __init__(self):
            self.choices = [Mock(message=Mock(content="Generated Documentation"))]

    return MockResponse()

def test_generate_documentation2():
    """
    Test the generate_documentation function using a mocked successful OpenAI API response.

    This test function uses the @patch decorator to replace the call to openai.ChatCompletion.create
    with a mock response generated by the mock_openai_response() function. It then tests the
    generate_documentation function to ensure it correctly interacts with the OpenAI API and returns
    the expected documentation content.

    Test Steps:
    1. Replace the OpenAI API call with a mock response using the @patch decorator.
    2. Define a prompt and source code for documentation generation.
    3. Call generate_documentation with the prompt and source code.
    4. Assert that the generated documentation contains the expected content.

    """

    with patch("openai.ChatCompletion.create", return_value=mock_openai_response()):
        prompt = "Generate Python documentation for a function"
        source_code = """
def greet(name):
    '''This function greets the given name.'''
    print('Hello,', name)
"""
        documentation = generate_documentation(prompt, source_code)
        assert "Generated Documentation" in documentation

def test_generate_documentation_api_error():
    """
    Test the generate_documentation function handling of an OpenAI API error.

    This test function uses the @patch decorator to replace the call to openai.ChatCompletion.create
    with a side effect that raises an Exception, simulating an API error. It then tests the
    generate_documentation function to ensure it properly handles the exception.

    Test Steps:
    1. Replace the OpenAI API call with a mock exception-raising side effect using the @patch decorator.
    2. Define a prompt and source code for documentation generation.
    3. Use pytest's raises context manager to expect an Exception.
    4. Call generate_documentation with the prompt and source code, which should raise the expected exception.
    5. Assert that the raised exception's message matches the expected error message.

    """

    with patch("openai.ChatCompletion.create", side_effect=Exception("API Error")):
        with pytest.raises(Exception) as e_info:
            prompt = "Generate Python documentation for a function"
            source_code = """
def greet(name):
    '''This function greets the given name.'''
    print('Hello,', name)
"""
            generate_documentation(prompt, source_code)
        assert str(e_info.value) == "API Error"


def test_main_function():
    """
    Test the main functionality of the main() function for generating documentation.

    This test function simulates the main functionality of the `main()` function by
    mocking various dependencies. It ensures that the `main()` function correctly
    interacts with the mocked objects, performs file operations, and generates
    documentation.

    Steps:
    1. Create mock objects for argparse arguments, parser, open function, and generate_documentation function.
    2. Configure mock objects with appropriate behavior and return values.
    3. Use patches to replace built-in functions and modules with mock objects.
    4. Execute the main() function within a controlled context.
    5. Perform assertions to verify the interactions and calls made during the execution.

    Assertions:
    - Ensure the argparse parser is called once to parse command-line arguments.
    - Verify the 'open' function is called with specific arguments.
    - Confirm the 'read' and 'write' methods are called on the mock open object.
    - Validate that the 'generate_documentation' function is called with specific arguments.

    """
 
    mock_args = Mock()
    mock_args.source_file = "source.sas"
    mock_args.documentation_file = "documentation.txt"
    mock_args.prompt = "test prompt"

    mock_parser = Mock()
    mock_parser.parse_args.return_value = mock_args

    mock_open = mock_builtin_open()

    mock_source_file_content = "Mocked source file content"
    mock_open().__enter__().read.return_value = mock_source_file_content

    mock_generate_documentation = Mock(return_value="Mocked documentation")

    with patch("argparse.ArgumentParser", return_value=mock_parser), \
         patch("builtins.open", mock_open), \
         patch("generate_documentation.generate_documentation", mock_generate_documentation):

        main()

    # check parse_args call
    mock_parser.parse_args.assert_called_once()
    
    # check open calls
    mock_open.assert_has_calls(
        [
            call("source.sas", "r"),
            call("documentation.txt", "w")
        ],
        any_order=True,
    )
    
    # check read and write calls
    mock_open().__enter__().read.assert_called_once()
    mock_open().__enter__().write.assert_called_once_with("Mocked documentation")

    # check generate_documentation call
    mock_generate_documentation.assert_called_once_with("test prompt", mock_source_file_content)



def test_main_function_error():
    """
    Test the error handling behavior of the main function when generating documentation.

    This test function simulates an error scenario in the `main()` function by
    mocking various dependencies. It checks that the `main()` function handles
    exceptions properly and interacts with mocked objects as expected.

    Steps:
    1. Create mock objects for argparse arguments, parser, open function, and generate_documentation function.
    2. Configure mock objects with appropriate behavior and side effects.
    3. Use patches to replace built-in functions and modules with mock objects.
    4. Execute the main() function within a pytest context manager that expects an exception.
    5. Perform assertions to verify the interactions and calls made during the execution.

    Assertions:
    - Ensure the argparse parser is called once to parse command-line arguments.
    - Verify the 'open' function is called with specific arguments.
    - Confirm the 'read' method is called on the mock open object.
    - Validate that the 'generate_documentation' function is called with specific arguments.

    """

    mock_args = Mock()
    mock_args.source_file = "source.sas"
    mock_args.documentation_file = "documentation.txt"
    mock_args.prompt = "test prompt"

    mock_parser = Mock()
    mock_parser.parse_args.return_value = mock_args

    mock_open = mock_builtin_open()

    mock_source_file_content = "Mocked source file content"
    mock_open().__enter__().read.return_value = mock_source_file_content

    mock_generate_documentation = Mock(side_effect=Exception("Mocked error"))

    with patch("argparse.ArgumentParser", return_value=mock_parser), \
         patch("builtins.open", mock_open), \
         patch("generate_documentation.generate_documentation", mock_generate_documentation):

        with pytest.raises(Exception, match="Mocked error"):
            main()


    mock_parser.parse_args.assert_called_once()

    # check open calls
    mock_open.assert_has_calls(
        [
            call("source.sas", "r"),
        ],
        any_order=True,
    )
    # check read call
    mock_open().__enter__().read.assert_called_once()

    # check generate_documentation call
    mock_generate_documentation.assert_called_once_with("test prompt", mock_source_file_content)
