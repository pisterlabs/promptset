# This tells your code to use the OpenAI API library
import openai

# Change this to your personal API key
openai.api_key = '<ENTER-YOUR-API-KEY-HERE>'

# Edit the text below to customize the input for GPT-3's text generator
my_prompt = "Give me two animals that are cartoon characters."

# This is the primary function that uses GPT-3 to generate a response.
def generate_response(prompt):

    # This code specifies how we will be using GPT-3.
    # Changing the parameters below will give you different results.
    # This function uses the Completion feature from the OpenAI API library
    # For more details, view the OpenAI API documentation at https://beta.openai.com/docs/api-reference/introduction
    response = openai.Completion.create(

        # This is the text generator model.
        # Options are: text-davinci-001, text-curie-001, text-babbage-001, text-ada-001
        # text-davinci-001 is the most powerful, but also the slowest
        engine="text-davinci-001",

       # This is the prompt (as a string) that GPT-3 will complete
       prompt=prompt,

       # Temperature (0.0 to 1.0) controls the variation of the responses.
       # A temperature of 0 will give you the same response every time you run the code,
       # and a temperature of 1 will give you a wide range of responses.
       temperature=0.9,

       # Tokens are how GPT-3 breaks up the text into smaller sub-units.
       # Tokens can be words, portions of words, or punctuation.
       # max_tokens (0 to 2048) determines the the maximum length of the response.
       max_tokens=50,

    )

    # The output of the function is the response GPT-3 generates
    return response

# Call your function and print out the response.
print(generate_response(my_prompt).choices[0].text)

# Challenge 1:
# Write a function, multiple_responses, that comes up with n *unique* responses
# to prompt p.

# Tip: What checks will you have to do to make sure you have n unique responses and no repeats?

# Example input: p = 'My favorite thing about Whitman College' and n = 3
# Example output:
# '''Response 1:   is the excellent faculty. The professors are passionate about teaching and are always willing
# Response 2: My favorite thing about Whitman College is the campus community. Whitman students
# Response 3:  is that is provides a very welcoming and inclusive community for students of all backgrounds'''

# Write your function below.
#def multiple_responses(p,n):

# Challenge 2:
# Write a function, GPT3_chain, that uses an initial prompt, p, then uses the response it generates
# as a *new* prompt to generate another response... and does this n times.

# Example input: p = 'Walla Walla is' and n = 3
# Example output: '''Prompt: Walla Walla is
# Response 1: known for its up and coming wine industry. There are over 130
# Response 2:  wineries in the Willamette Valley, most of which are located in
# Response 3: the northern and eastern parts of the valley. The valley is home
# Response 4: to a number of endangered species, including the California condor, the Sierra'''

# Write your function below.
#def GPT_chain(p,n):

# Challenge 3:
# The generate_response function above includes the parameters, temperature, engine, and max_tokens,
# but there are more built parameters into the GPT-3 OpenAI API.

# Go to the documentation at https://beta.openai.com/docs/api-reference/completions/create?lang=python
# and pick one of the additional parameters (top_p, presence_penalty, or frequency_penalty).

# Modify the generate_response function to include one of these additional parameters.

# Then, write a new function, parameter_test, that returns the response over a range of values
# of these parameters (i.e., for top_p, 0.1, 0.2, 0.3,...0.9, 1.0 ) for a single prompt, p

# TIP: Recommend setting the temperature to 0 for this.

# Write your function below.
#def parameter_test():

# Challenge 4:
# For each of the new functions that you wrote (multiple_responses, GPT3_chain, parameter_test),
# Write one *unit test* to make sure it is performing as intended.
