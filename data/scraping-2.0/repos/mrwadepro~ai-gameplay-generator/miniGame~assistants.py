from openai import OpenAI
import os
import time
import re

testKey = os.environ.get("API_KEY")
# in your terminal, please add export API_KEY=<api key>
def extract_and_save(text, filename):
    # Use a regular expression to find text between triple backticks
    match = re.search(r'```(?:Python|python)(.*?)```', text, re.DOTALL)
    
    if match:
        extracted_text = match.group(1).strip()
        pattern = r'\\n'

# Replace '\\n' with '\n' to format the string correctly
        converted_code = re.sub(pattern, '\n', extracted_text)
        # Save the extracted text to a file
        with open(filename, 'w') as file:
            file.write(converted_code)
    else:
        print("No triple backticks found in the input text.")

client = OpenAI()
game_designer = client.beta.assistants.retrieve("asst_ppZYfGoyh6ISVvD5XyLI6Ayu")
game_developer = client.beta.assistants.retrieve("asst_WgenLJkZXnbFoIfFzpB80eiW")
software_developer = client.beta.assistants.retrieve("asst_TzwMLh8i8hnVeqDMl9iLhhd6")
code_reviewer = client.beta.assistants.retrieve("asst_bYXkG2Gy5i1s2lk3d6qpfU1C")

userInput = input("Please enter in your initial learning standard")
main_thread = client.beta.threads.create()
thread_message = client.beta.threads.messages.create(
  main_thread.id,
  role="user",
  content="Please design a simple mini game for this standard: " + userInput,
)

run = client.beta.threads.runs.create(
  thread_id= main_thread.id,
  assistant_id= game_designer.id
)

waitTime = 3
t = 0
while (run.status != "completed"):
    run = client.beta.threads.runs.retrieve(
        thread_id= main_thread.id,
        run_id= run.id
    )
    
    print(t, "game designer ", run.status)
    time.sleep(waitTime)
    t+= waitTime
    
run = client.beta.threads.runs.create(
  thread_id= main_thread.id,
  assistant_id= game_developer.id
)

while (run.status != "completed"):
    run = client.beta.threads.runs.retrieve(
        thread_id= main_thread.id,
        run_id= run.id
    )
    print(t , "game developer ",run.status)
    time.sleep(waitTime)
    t+= waitTime


#print(run)
thread_messages = client.beta.threads.messages.list(main_thread.id)
print(thread_messages.data[0].content)
fileName = input("please enter game file name")
extract_and_save(str(thread_messages.data[0].content[0]), fileName)






#thread = openai.
