import os
import openai

try:
    openai_api_key=os.getenv("OPENAI_API_KEY");
except Exception as e:
    print("ERROR: OPENAI_API_KEY environment variable not set, you cannot use OpenAI API functions without this key.")
    print("...check with your administrator for assistance procuring or registering the key.")
    exit(1) 

print(f"Registering the Open API key with the library.")
print("")
openai.api_key = openai_api_key
#print(openai.Model.list())

print(f"Initiating simple ChatCompletion with Hello World prompt.")
try:
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                              messages=[{"role": "user", "content": "Hello world"}])
except Exception as e:
    print("ERROR: openai.ChatCompletion failed.")
    print("...the following error resulted from teh API call:")
    print(f"......{str(e)}")
    exit(1) 



print("Response:")
print("##################################################################")
print(completion.choices[0].message.content)
print("##################################################################")
