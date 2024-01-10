from openai import OpenAI
import json
import os

client = OpenAI(api_key="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")

def read_agent_metadata(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def agent_daily_plan_chat(agent_details):
    prompt = f'''
    You are a virtual assistant creating a unique daily plan for agent {agent_details["Name"]} who works in a 
    University. Their role is {agent_details["Role"]}. The plan should reflect what is typical 
    of an ordinary working day from 09:00 to 05:00 with lunch between 12:00 - 13:00 and 
    breaks throughout for meetings etc. Be very specific and to the point, do not add too much description. The plan
    should very easy to read and understand. Attach (T) or (B) or (M) in front of each distinct time step containing 
    activities. (T) is task, (B) is break, and (M) is meeting. This is to support parsing. 
    Provide the output in JSON format. Activities are in plan, and time, activity, and type should be nested in plan.

    Name: {agent_details["Name"]}
    Age: {agent_details["Age"]}
    Role: {agent_details["Role"]}
    Gender: {agent_details["Gender"]}
    

    '''

    response = client.chat.completions.create(model="gpt-3.5-turbo",  # or the most appropriate model you have access to
                                              messages=[
                                                  {"role": "system", "content": prompt}
                                              ])

    response_message = response.choices[0].message.content
    print(response_message)

    #return response["choices"][0]["content"]  # Access the content directly
    #return response["choices"][0]["message"]["content"]

def generate_daily_plan(agent):
    return agent_daily_plan_chat(agent)


def main():
    # Reading agent metadata from the file
    agents_metadata = read_agent_metadata('/Users/lewnew/PycharmProjects/SANDMAN/src/agents/agent_metadata.json')

    # Ensuring to process only 5 agents if there are more
    agents_metadata = agents_metadata[:1]

    # Generating and printing plans for each agent
    for agent in agents_metadata:

        plan = generate_daily_plan(agent)

        # Print the plan to the terminal
        print(f'Plan for {agent["Name"]}:')
        #print(plan)
        #print('---------------------------------------------')

        # Save the plan to a JSON file
        plan_filename = f'plan_{agent["Name"].replace(" ", "_")}.json'
        with open(plan_filename, 'w') as json_file:
            json.dump({"plan": plan}, json_file, indent=4)
            print(f'Plan for {agent["Name"]} has been saved to {plan_filename}')


if __name__ == '__main__':
    main()
