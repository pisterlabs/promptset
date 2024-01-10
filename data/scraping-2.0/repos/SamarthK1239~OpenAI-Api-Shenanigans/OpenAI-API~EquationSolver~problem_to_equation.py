# Using a class proved necessary in this case, as the OpenAI API requires the API key and organization key to be set as environment variables.
class ChatGPT:
    # Same old imports
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    import openai

    # Load the environment variables
    path = Path("EquationSolver/Environment-Variables/.env")
    load_dotenv(dotenv_path=path)

    # Don't really need this, might remove it later
    ORGKEY = os.getenv('organization')
    APIKEY = os.getenv("api_key")

    # Initialize the class
    def __init__(self):
        self.openai.organization = self.ORGKEY
        self.openai.api_key = self.APIKEY

    # Function to convert the problem to an equation (switched from davinci-003 to gpt-3.5-turbo)
    # Prompt design is especially important here
    def convertProblemToEquation(self):
        word_problem = input("Enter a word problem: ")
        response = self.openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "Use the word problem from below to create an arithmetic equation(s), using any numerical figures from the question. You may create multiple equations if required to answer the question."
                               "Respond with only mathematical equation(s) and no text whatsoever. Ensure that the equation(s) you provide can be directly entered into a tool like "
                               "symbolab to obtain an answer. Include brackets wherever needed for clarity. \n" + word_problem
                }
            ],
            # prompt="Use the word problem from below to create an equation, using any numerical figures from the question. Respond with only a mathematical equation and no text whatsoever. I do not need any explanatory text accompanying the equation. \n" + word_problem,
            temperature=0.3,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        return response["choices"][0]["message"]["content"]

    # Deprecated function, kept for reference to previous versions ONLY
    # Don't use this, it's not very good, and the model referenced is deprecated lol
    def extractEquation(self, response):
        equation = self.openai.Completion.create(
            model="text-davinci-003",
            prompt="From this text, extract an equation which i can put into an equation solver such as symbolab, and respond with only the equation and no accompanying text: \n" + response,
            temperature=0.3,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        return equation["choices"][0]["text"]
