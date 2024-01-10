
import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

def generate_todolist_output():
    # Your end goal and other necessary data
    end_goal = '''1. health insurance
                2. leetcode
                3. todolist side project
                4. start resume making
                5. 10am-11:30am - meeting'''
    time_limit = "24 hours"
    # Other inputs as needed

    prompt = '''You're an assistant designed to create a plan to achieve an end goal within a given time limit. 
            The goal should be broken down into tasks and their timeline. The user might provide the timeline as 
            days, weeks, months, or an end date. If an end date is given, calculate the time remaining based on the current date. 
            You may also act as a planner, breaking down the user's daily tasks into smaller ones for better 
            completion due to possible ADHD. If a list of tasks is given, break tasks down into subtasks or elaborate on those tasks.
             Create a gamified to-do list and also schedule breaks to make tasks fun and avoid burnout.
            Suggest methods to relax etc during those break times. Also allocate time for daily activities like lunch, showering etc.
            If asked to plan their day, start from 9 am, allocate specific times for each task, and consider predetermined 
            time periods for certain tasks. Your response is always just JSON which looks like this example structure:
            {
                "time": {{insert total time}},
                "tasks": [
                    {"time": {{insert time period 1 e.g 9am-10am}}, "task": {{insert task 1}}},
                    {"time": {{insert time period 2 e.g 11am-2pm}}, "task":  {{insert task 2}}}
                ]
            }
            Remember to only give your output in this format and schedule the tasks keeping in mind user's scheduling restrictions '''
    user_msg = f"My tasks are: {end_goal} and my time limit: {time_limit}"

    # Make the API call
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",  # Use the GPT-3.5 engine
        messages = [{
            "role":"system",
            "content":prompt
        },
        {
            "role":"user",
            "content": user_msg
        }],
        max_tokens=1024,  # Adjust this based on the length of the response you expect
        temperature=0.5,  # Controls the randomness of the output. 0 means deterministic, 1 means very random.
        n=1,  # Number of completions to generate
        stop=None  # Stop generating after a certain token. You can use stop to guide the response length.
    )
    # return status code
    status = response["choices"][0]["finish_reason"]

    generated_reply = response["choices"][0]["message"]["content"]
    # reply2 =  response["choices"][1]["message"]["content"]
    tokens_used = len(prompt)+len(user_msg)+len(generated_reply)
    tokens_used = tokens_used/4
    print("Tokens used: ", tokens_used)
    print("response 1", generated_reply)
    # print("response 2", reply2)
    # display(Markdown(generated_reply))
    return generated_reply

if __name__ == '__main__':
    generate_todolist_output()
