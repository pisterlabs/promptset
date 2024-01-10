import openai
import requests
import os
import time

"""
Variables that requires user inputs will be marked with TODO
"""
# light models
# FINETUNED_MODEL = "ada:ft-personal-2023-06-01-09-57-07"
# FINETUNED_MODEL = "ada:ft-personal-2023-06-05-08-24-08"
# FINETUNED_MODEL = "ada:ft-personal-2023-06-05-08-58-58" # latest

# pramukas model
# FINETUNED_MODEL = "ada:ft-personal-2023-07-05-11-03-17" # train 300
# FINETUNED_MODEL = "ada:ft-personal-2023-07-10-12-55-09" # train 252
FINETUNED_MODEL = "ada:ft-personal-2023-07-10-20-32-27" # train 504

TEST_DATA_FILE = "data/test_data.txt"
ANSWER = 1
# number of times to call api
NUM_TIMES = 1

# openAI api request
def httpRequest(prompt):
    
    url = 'https://api.openai.com/v1/completions'
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=prompt).json()
    try:
        # cases
        ONE_TAP  = "SingleTap"
        TWO_TAPS = "DoubleTap"
        HOLD = "Hold"

        print("Response:")
        generated_text = response["choices"][0]["text"]
        # print("generated_text: ", generated_text)

        if ONE_TAP in generated_text:
            return 1
        elif TWO_TAPS in generated_text:
            return 2
        elif HOLD in generated_text:
            return "Hold"
        else:
            print(f"UNKNOWN: {generated_text}")
            return "Unknown"
        # return generated_text
    except:
        print(f"Request failed with status code {response['error']}")


def main():

    gesture_count=1
    # Load file
    filename3 = TEST_DATA_FILE

    # clear file
    with open(filename3, 'w') as f:
        f.write("")
    f.close()
    while True:

        # wait for file to be updated
        initial_modification_time = os.path.getmtime(TEST_DATA_FILE)
        while True:
            # Check the current modification time of the file
            current_modification_time = os.path.getmtime(TEST_DATA_FILE)
            # Compare the modification times
            if current_modification_time > initial_modification_time:
                # prompt_data = "Data:\n"
                # prompt_data=""
                with open(filename3, 'r') as f:
                    prompt_data = f.read()
                # prompt_data+="\n\nAnswer:"
                # f.close()
                print("prompt_data: ", prompt_data)
                break


        start_time = time.time()
        correct_ans = 0


        
            
        for i in range(NUM_TIMES):

            # edit here
            prompt = {
                "prompt":prompt_data,
                # change model when it is updated
                "model": FINETUNED_MODEL,
                "temperature":0.3
            }

            print("Prompt {} has been sent. Waiting for response...".format(i+1))
            apiResponse = httpRequest(prompt)

            if apiResponse == ANSWER:
                correct_ans += 1

            if apiResponse=="Hold":
                print(f"Gesture {gesture_count}: Hold detected \n")
            else:
                print(f"Gesture {gesture_count}: {apiResponse} tap(s) detected \n")

        end_time = time.time()
        gesture_count+=1
        # print(end_time-start_time,"s has elapsed")
        # print(f"Gesture: {ANSWER} taps | Score: {correct_ans}/{NUM_TIMES} tap(s) correct | Total Time: {end_time-start_time}s")

main()