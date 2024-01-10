import openai
import tiktoken
import constants

def auth():
    openai.api_key = constants.OpenAPI_Key

def openai_response():
    response = openai.Completion.create(
        model = "text-davinci-003",
        prompt = "Can you write me a quick python script that uses a while loop and prints 1 to 10? When you return an output, can you only give me the code without any dialogue or comments.",
        temperature = 0,
        max_tokens = 1000,
        stream = True,
    )
    return response

def compare(response):
    response = str(response)
    work_file = open("newfile.py", "r")
    new_string = response.strip()
    if (str(new_string) == str(work_file.read())):
        print("Files match")
    else:
        print(str(new_string))
        print(str(work_file))
        print("Files do not match.")
    

def streamAndPrint(response):
    completion_text = ''
    for event in response:
        event_text = event['choices'][0]['text']
        completion_text += event_text
    return completion_text

if __name__ == "__main__":
    auth()
    response = openai_response()

    print_response = streamAndPrint(response)
    print(f"Full text received: {print_response}")
    print_response_2 = str(f"{print_response}")

    compare(print_response_2)
