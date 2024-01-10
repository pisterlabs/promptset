import openai
import requests
import json
import tiktoken
import string
import os

"""This class initilizes a GPT4 engine to query the GraphQL API"""
class GraphQLGPTEngine:
    def __init__(self):
        self.user_input = ""
        self.model_name = 'gpt-4'
        self.encoding = tiktoken.encoding_for_model(self.model_name)
        self.schema = ""
        self.instruction_prompt = "For the following statement, please generate the GraphQL query code ONLY. No explanation."
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.query_string = ""
        self.base_url = "https://api.platform.opentargets.org/api/v4/graphql"
        self.api_response = ""
        self.final_result = ""

    """This method reads user input"""
    def get_user_input(self):
        self.user_input = input("Please enter your question in English: \n")

    """This method loads the full schema of graphql from here: """
    def load_graphql_schema(self):
        try: 
            response = requests.get(self.base_url+"/schema")
            self.schema = "#Full graphql schema:\n\n"+response.text
        except requests.exceptions.HTTPError as err:
            print(err)
        
    """This method checks the token length which is used later during model initialization"""
    def get_token_length(self):
        token_length = len(self.encoding.encode(self.schema))+len(
            self.encoding.encode(self.instruction_prompt))+len(
            self.encoding.encode(self.user_input))
        return token_length

    """This method generates the quert string"""
    def generate_query_string(self):
        openai.api_key = self.api_key
        
        #Get the token length
        token_length = self.get_token_length()
        
        #Check if the token length is supported by the OpenAI GPT4 model,
        #else reduce the question size to fit the appropriate token length
        while token_length>=8192: 
            print("\nReduce the size of your question.")
            self.user_input =  input("Please re-enter your question in English: \n")
            token_length = self.get_token_length()
            
        #Initializes the messages
        messages_array = [
            {"role": "system", "content": self.schema},
            {"role": "system", "content": self.instruction_prompt},
            {"role": "user", "content": self.user_input}
        ]    
        
        #Get the response from GPT4 model
        response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages_array,
                temperature=0,
                max_tokens=250,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["###"]
            )

        #Get the Query string from the response
        self.query_string = response['choices'][0]['message']['content']
        
    """This method performs the GraphQL api request to get the response"""
    def perform_api_request(self):
        try:
            #Set the generated query string to the API request
            response = requests.post(self.base_url, json={"query": self.query_string})
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(err)

        #assign the response from the API
        self.api_response = json.loads(response.text)

    """This method mines the GraphQL API response using GPT-4 to generate the final result as requested"""
    def generate_final_result(self):
        
        #This prompt is to mine the results from the response
        instruction_prompt = "The following text is the response of the request: " + self.user_input + ".\n The final answer should just list the queried entities, no extra paragraphs or text"
        messages_array = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": str(self.api_response)}
        ]

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages_array,
            temperature=0,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["###"]
        )

        self.final_result = response['choices'][0]['message']['content']
        print(self.final_result)


if __name__ == "__main__":
    
    processor = GraphQLGPTEngine()
    processor.get_user_input()
        
#     print("Loading Schema...")
    processor.load_graphql_schema()
    
#     print("Generating Query String...")
    processor.generate_query_string()
    
#     print("Querying API...")
    processor.perform_api_request()
    
#     print("Generating Results... ")
    processor.generate_final_result()
