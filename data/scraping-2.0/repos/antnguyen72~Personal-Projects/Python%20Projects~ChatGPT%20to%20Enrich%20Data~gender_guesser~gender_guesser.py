import openai
import time
import json

def call_openai(prompt: str) -> str:
    """
    Call the OpenAI API and return the generated text.

    Args:
        prompt (str): The prompt to generate text from.

    Returns:
        str: The generated text.
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0]['text'].strip()
    except openai.error.RateLimitError:
        # Catch the rate limit error and wait for one minute before trying again
        print("Rate limit reached. Waiting for 60 seconds before trying again...")
        time.sleep(60)
    except Exception as e:
        # Catch any exceptions thrown by the OpenAI API and print the error message
        print(f"Error calling OpenAI API: \nError Type: {type(e)}\nError Message: {e}")
        return "Error calling OpenAI API"


class GenderGuesser:
    def __init__(self, openai_apikey: str, gender_guesser_data_file_name: str):
        self.data_dict = {}
        self.openai_apikey = openai_apikey
        self.gender_guesser_data_file_name = gender_guesser_data_file_name

        # keeps track of the number of calls made
        self.total_count = 0

        # Will make it so that the file is saved every 100 calls
        self.count = 0

        # init api
        openai.api_key = openai_apikey

    def intake_data(self):
        with open(self.gender_guesser_data_file_name,'r') as file:
            self.data_dict = json.load(file)
    
    def save_data(self):
        with open(self.gender_guesser_data_file_name,'w') as file:
            json.dump(self.data_dict,file)

    def gender_guesser(self, name: str)-> str:
        if self.count == 100:
            print("\n100 names has been processed...saving before continuing\n")
            self.save_data()
            self.count = 0
        name = str(name)
        if name in self.data_dict:
            return self.data_dict[name]
        else:
            print(f"--{name}-- is not found in the databse. Using chatgpt...")
            prompt = f"""
            Guess the gender of someone named {name}
            If Male, return M
            If Female, return F
            If Androgenous, return A
            """
            gender = call_openai(prompt)
            print(f"{name} is suggested to be of gender = {gender}\n")
            self.data_dict[name] = gender
            self.count+=1
            self.total_count+=1

            return gender