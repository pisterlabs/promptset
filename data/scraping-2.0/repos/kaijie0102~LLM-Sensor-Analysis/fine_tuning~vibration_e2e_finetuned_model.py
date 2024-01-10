import openai
import requests
import os
import time
import sys

# print(sys.path)

"""
Variables that requires user inputs will be marked with TODO
"""
# FINETUNED_MODEL = "ada:ft-personal-2023-06-01-09-57-07"
# FINETUNED_MODEL = "ada:ft-personal-2023-06-05-08-24-08"
FINETUNED_MODEL = "ada:ft-personal-2023-06-05-08-58-58" # vibration 30 1 tap, 30 2taps, 30 3 taps
TEST_DATA_FILE = "data/vibration_raw_data.txt"

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
        ONE_TAP  = "one tap"
        TWO_TAPS = "two taps"
        THREE_TAPS = "three taps"

        print("\nResponse:")
        generated_text = response["choices"][0]["text"]
        # print("generated_text: ", generated_text)

        if ONE_TAP in generated_text:
            return 1
        elif TWO_TAPS in generated_text:
            return 2
        elif THREE_TAPS in generated_text:
            return 3
        else:
            return "Unknown"
        # return generated_text
    except:
        print(f"Request failed with status code {response['error']}")

def in_range(val,lower, upper):
    if val<lower or val>upper:
        return False
    return True

def extract_lines(filename):
    THRESHOLD = 0.07
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # pre processing
        for index in range(len(lines)):
            # print("current: ", lines[index])
            if (len(lines[index])>20):
                # print("overlength, trim! ", lines[index])
                lines[index] = lines[index][15:]
                # print("aftertrim: ", lines[index])
            lines[index] = lines[index].split() # to make line an row of 3 elements

        # print(lines)

        # setting lower and upper bound
        lines[0][2].strip()
        upper_bound = float(lines[0][2]) + THRESHOLD
        lower_bound = float(lines[0][2]) - THRESHOLD
        tap_started = False
        end_count=0
        start_index=-1
        end_index=-1

        for i in range(len(lines)):
        
            # find start index
            # print("current before: ", float(lines[i][2]))
            current = float(lines[i][2].strip())
            if not in_range(current,lower_bound,upper_bound) and not tap_started:
                start_index = i
                tap_started = True
            
            elif tap_started:
                if in_range(current,lower_bound,upper_bound):
                    end_count+=1
                else: 
                    end_count=0

                if end_count>20:
                    # tap has ended
                    end_index = i - 20
                    break

    # start_index=5
    # end_index=82
    if start_index==-1 or end_index==-1:
        return -1
    else:
        test_data = lines[start_index-5:end_index+5]
        # print("got new line: ", test_data)
        return test_data
    


def main():
    total_count = 0
    tap_correct = 0 

    # wait for file to be updated
    # while True:
    timeout = 2
    start = False
    last_modified_time = 0

    # Start an infinite loop
    while True:
        # Sleep for a short duration to avoid excessive CPU usage
        # time.sleep(1)


        print("Waiting for data from sensor...")
        initial_modification_time = os.path.getmtime(TEST_DATA_FILE)
        while True:
            # Check the current modification time of the file
            current_modification_time = os.path.getmtime(TEST_DATA_FILE)
            last_modified_time = current_modification_time
            # Compare the modification times
            if current_modification_time > initial_modification_time and not start:
                print("Sensor data is coming in")
                start = True
                # break

            # check that it has been unmodified for 2 secnods
            elif start:
                # File hasn't been updated
                time.sleep(timeout)                
                # Check if the file remains unchanged after the timeout
                if os.path.getmtime(TEST_DATA_FILE) == last_modified_time:
                    print(f"All data sent")
                    start=False
                    break
    


        # Print a message indicating that the file is still not updated
        # print("Waiting for the file to be updated...")
    
        # extract relevant rows of data for testing
        test_data = extract_lines(TEST_DATA_FILE)
        if test_data==-1:
            print("No Taps Detected")
            break
        
        # print("test_data: ", test_data)

        # number of times to call api
        num_times = 1
        start_time = time.time()
        for i in range(num_times):
            # Load mqtt file
            # filename3 = TEST_DATA_FILE
            prompt = "Data:\n"
            # with open(filename3, 'r') as f:
                # prompt += f.read()
            for row in test_data:
                for element in row:
                    prompt+=element
                    prompt+=" "
                prompt+="\n"
            prompt+="\n\nAnswer:"

            
            # print("Sending Prompt: \n"+prompt)

            # edit here
            prompt = {
                "prompt":prompt,
                # change model when it is updated
                "model": FINETUNED_MODEL,
                "temperature":0.3
            }

            print("Prompt {} has been sent. Waiting for response...".format(i+1))
            apiResponse = httpRequest(prompt)

            if apiResponse==None:
                print("Context too long")
            else:   
                print("{} tap(s) detected".format(apiResponse))

        end_time = time.time()
        print("Length of data: ", len(test_data))
        print(end_time-start_time,"s has elapsed")
        total_count+=1
        print()
        # print("Tested: ",total_count,". Correct: ",tap_correct)

main()
