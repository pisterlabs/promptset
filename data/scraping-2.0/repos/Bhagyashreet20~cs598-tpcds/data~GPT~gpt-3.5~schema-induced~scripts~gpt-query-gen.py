import os
import openai
import time
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def llm(model,message):
    max_retries = 3  # Maximum number of retries
    retries = 0

    while retries < max_retries:
        try:
            start_time = time.time()  # Start time
            output = openai.ChatCompletion.create(model=model, messages=message)
            end_time = time.time()  # End time

            time_taken = end_time - start_time  # Calculate time taken
            query = output["choices"][0]["message"]["content"]

            return (time_taken, query)  # Return the result if successful
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")
            retries += 1
            time.sleep(60)  # Sleep for 1 minute before retrying

    print("Max retries reached. Unable to complete the operation.")
    return None

def main():
   
    for i in range(1,100):
        prompt_path = os.path.abspath(f"../prompts/oneshot-prompt{i}.txt")
        response_time_path = os.path.abspath(f"../llm-gen-time/schema-induced-gen-time{i}.txt")
        query_path = os.path.abspath(f"../queries/schema-induced-query{i}.sql")

        with open(prompt_path,"r") as f:
            prompt = f.read()
        prompt = json.loads(prompt)

        time_taken, query = llm("gpt-3.5-turbo",prompt)
        
        with open(response_time_path,"w") as f:
            f.write(str(time_taken))
        with open(query_path,"w") as f:
            f.write(query)
       
        
        time.sleep(60) #45s time delay to ensure that the API isn't throttled

if __name__ == "__main__":
    main()
