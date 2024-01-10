##############################################################
#   This is a very simple demonstrator, that shows how 
#   HashiCorp Vault can be used to store API keys used 
#   in a Python application.
#   The application simulates a simple GPT-3.5-Turbo based
#   ChatBot with a ridumentary Google search function.
##############################################################

import hvac   
import openai
import requests

##############################################################
#   HashiCorp Vault Initialization and retrieval of API keys 
##############################################################
class Keychain:
    def __init__(self):
        client = hvac.Client(
            url='https://vault-int.app.corpintra.net/',
            namespace='HCV-test'
        )
        approle_file = open("approle.txt", "r")
        approle_secret_id = approle_file.readline()
        approle_file.close()
        client.auth.approle.login(
            role_id='8c9d6f48-f20e-93b6-aff4-b9e1e15b4e78',
            secret_id=approle_secret_id
        )
        read_response = client.secrets.kv.read_secret_version(path='chatbot', mount_point='kv')        
        self.google_search_api_key = read_response['data']['data']['google_search_api_key']
        self.openai_api_key = read_response['data']['data']['openai_api_key']

    def get_google_search_api_key(self):
        return self.google_search_api_key
    
    def get_openai_api_key(self):
        return self.openai_api_key
        

##############################################################
#   ChatBot
#   requires a Keychain object which stores API keys
#   and True if you want to add Google search results
##############################################################
class ChatBot:
    def __init__(self, keychain, use_google_search):
        # the initial context of the chat, you can change it to tune the personality of the assistant.
        openai.api_key = keychain.get_openai_api_key()
        self.google_search_api_key = keychain.get_google_search_api_key()
        self.chat_messages = [{ "role" : "system", "content" : "You are a helpful assistant."}]
        self.use_google_search = use_google_search

    # generate response based on the current chat messages
    # it starts with the initial context and gets expanded by user inputs and generated responses
    # see https://platform.openai.com/docs/api-reference/chat/create
    def __generate_response(self):
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",        # used model
            messages = self.chat_messages,  # the array of messages
            max_tokens = 1000,              # max value of gpt-3.5-turbo is 4096
            temperature = 0.5,              # 0.0 = correct, 1.0 = creative
            n = 1                           # number of choices generated for each input message
        )
        # the generated json is trimmed to include the content of the message only
        return completion.choices[0].message.content    
    
    # The actual conversation starts here
    # The user input and the generated chatbot output are added sequentially to the chat context
    # If Google search is used, a query is generated from the user input,
    # and the search results are added the the generated output
    def startConversation(self):
        print("\nYou speak to a ChatBot powered by GPT-3.5-Turbo, enter ABORT to terminate.")
        while True:
            user_input = input("\nUser: ")
            if user_input.lower() == "abort":
                print("Chat terminated.")
                break
            self.chat_messages.append({ "role" : "user", "content" : user_input})
            response = self.__generate_response()
            if (self.use_google_search):
                response = self.__add_google_search_results(user_input, response)
            self.chat_messages.append({ "role" : "assistant", "content" : response})
            print("\nChatBot: " + response)

    # performs simple google search based on user input and adds it to the generated response
    # invokes additional functions to perform google search
    def __add_google_search_results(self, user_input, response):
        search_query = self.__generate_search_query(user_input)
        search_results = self.__perform_google_search(search_query)
        for result in search_results:
            response = response + "\n\n" + result[0] + "\n" + result[1]
        return response
    
    # generates the google search query based on user input
    def __generate_search_query(self, user_input):
        prompt = "Please generate one google search query based on the following question:\n" + user_input
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",        
            messages = [{ "role" : "user", "content" : prompt}],  
            max_tokens = 1000,              
            temperature = 0.5,              
            n = 1                           
        )
        return completion.choices[0].message.content 
    
    # performs google search for the generated query, returns the first three results
    def __perform_google_search(self, query):
        SEARCH_ENGINE_ID = "43290867be9fa42f8"
        url = f"https://www.googleapis.com/customsearch/v1?key={self.google_search_api_key}&cx={SEARCH_ENGINE_ID}&q={query}"
        response = requests.get(url)
        search_results = response.json()["items"]
        result = []
        for entry in search_results[:3]:
            result.append([entry["title"], entry["link"]])
        return result
    
def main():
    keychain = Keychain()
    chatbot = ChatBot(keychain, True)
    chatbot.startConversation()

if __name__ == "__main__":
    main()