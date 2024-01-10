#The code uses OpenAI's GPT-3 API to generate a text-based representation of a flowchart based on a given task description. It prompts the user to input a task description, sends that prompt to the GPT-3 API, and receives a response containing the generated flowchart text representation, which is then printed as the output. Note that the API key needs to be set with a valid GPT-3 API key for the code to work.

import openai

# Set your OpenAI GPT-3 API key here
api_key = 'API_key'


def generate_flowchart(task_description):
    openai.api_key = api_key

    # Define the prompt for the GPT-3 API
    prompt = f"Build a flowchart for the given task:\n{task_description}\n\nFlowchart:"

    # Use the GPT-3 API to generate the text-based representation of the flowchart
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=3000,  # Adjust this value as needed to control response length
        # Stop generating after the first line break (end of flowchart)
        stop=["\n"]
    )

    # Extract the generated flowchart representation from the API response
    flowchart_representation = response['choices'][0]['text']

    return flowchart_representation


if __name__ == "__main__":
    task_description = input("Enter the task description:\n")
    flowchart_text_representation = generate_flowchart(task_description)
    print(flowchart_text_representation)
