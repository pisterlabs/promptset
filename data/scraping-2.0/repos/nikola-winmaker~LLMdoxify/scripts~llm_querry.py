import openai
import certifi
certifi.where()

# Set your OpenAI API key
openai.organization = "YOUR_ORGANIZATION"
openai.api_key = 'YOUR_KEY'


# Define a function to interact with the chat model
def LlmDoxify(function_code):
    # The `LlmDoxify` function is designed to generate a Doxygen-style comment for a given C function code.
    # It uses the OpenAI GPT-3 language model (engine: 'text-davinci-003') to generate the comment.

    # The function takes `function_code` as input, which is the C code for which a Doxygen comment is to be
    # generated. It constructs a `user_input` string by combining the prompt for the user and the `function_code`.
    # The prompt asks the user to write a Doxygen comment for the C function and provides some guidance on
    # including `@brief`, detailed description, `@param`, and `@return` tags.

    # After obtaining the response from GPT-3, the function extracts the generated comment from the response and
    # returns it as the result.

    user_input = "Write a doxygen comment for this C function:\n" + \
                 function_code + \
                 "Always include @brief and detailed description, @param and  @return. "
    response = openai.Completion.create(
        engine='text-davinci-003',  # Specify the engine
        prompt= 'User: ' + user_input + '\nAssistant:',
        max_tokens=200,  # Adjust as needed
        temperature=0.7,  # Controls randomness of the output, lower values make it more focused
        n=1,  # Specify the number of responses to generate
        stop=None  # Specify a stop sequence to end the response early (optional)
    )

    answer = response.choices[0].text.strip()
    return answer
