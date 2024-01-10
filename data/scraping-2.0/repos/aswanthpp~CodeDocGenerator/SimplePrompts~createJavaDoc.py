import sys
import traceback
import openai
import argparse

def get_opeai_key():
      with open('key.txt', 'r') as file:
        return file.read()
      
api_key = get_opeai_key()
openai.api_key = api_key

def getOpenAiCompletion(prompt):
    try:
        response = openai.Completion.create(
        engine="text-davinci-003",  # You can choose the most appropriate engine
        prompt=prompt,
        max_tokens=150,  # Adjust this based on the desired response length
        api_key=api_key
        )
        return response.choices[0].text
    except Exception as err:
        print(traceback.format_exc())
        return "Got Exception from Completion API"
    
def getOpenAiChatCompletion(prompt):
    try:
        response = openai.ChatCompletion.create(
        model="text-davinci-003",
        messages=[
            {"role":"user","content":"""Create Documentation file which contains folowwing sections for the given java file
                1. Identify functions
                2. Identify dependencies
                3. Initialization  
                4. Flow diagram"""
            },
            {"role":"assistant","content":"Certainly! However, please provide the Java file" },
            {"role":"user","content":f"Java File: \n {prompt}"}    
        ])

        return response['choices'][0]['message']['content']
    except Exception as err:
        print(traceback.format_exc())
        return "Got Exception from Completion API"    

def processInputFile(java_file_path):

    with open(java_file_path, 'r') as file:
        java_code = file.read()

    # prompt = f"Generate Javadoc comments for the following Java class and each functions and return the commented java class for the file :\n{java_code}"
    # prompt = f"""Create Documentation file which contains folowwing sections for the given file
    #             1. Identify functions
    #             2. Identify dependencies
    #             3. Initialization  
    #             4. Flow diagram
    #             Sample code:
    #             \n{java_code}"""
    
    prompt = f"""Create Software Requirement Specfication document for the below java
                Sample code:
                \n{java_code}"""
    print("--------------------------PROMPT--------------------------------------")
    print(prompt)
    print("----------------------------------------------------------------------")
    print("\n\n\n")
    
    # generated_comments= getOpenAiCompletion(prompt)
    # print("--------------------------RESPONSE------------------------------------")
    # print(generated_comments)
    # print("----------------------------------------------------------------------")


    # responseDoc=getOpenAiChatCompletion(java_code)
    # print(responseDoc)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--path', type=str, help="Java File Path", required=True)
        java_file_path = vars(parser.parse_args())['path']
        processInputFile(java_file_path)
    except Exception as err:
        print(traceback.format_exc())