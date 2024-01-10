import openai

openai.api_key = "YOUR API KEY"


def generate_test_cases_with_input(num_cases):
    screen_name = input(
        "Enter name of screen you are testing: ")
    screen_desc = input("Write a short description for your screen (e.g. This screen allows users...): ")
    elements = []
    print("Enter elements of the screen and type 'stop' when you are done:")
    while True:
        element = input("Enter element: ")
        if element.lower() == 'stop':
            break
        elements.append(element)
    feature_string = ", ".join(elements)
    prompt = f"Imagine you are a QA Engineer for the mobile app, and you have been given the task of testing the {screen_name} screen. {screen_desc} and has the following elements: {feature_string}. Write all possible test cases you can think of that would ensure the functionality of the {screen_name} screen and its elements are working correctly, including edge cases and negative test cases."
    test_cases_list = []
    for i in range(num_cases):
        response = openai.Completion.create(
            engine="text-davinci-003",
            temperature=0.94,
            max_tokens=3800,
            prompt=prompt
        )
        test_cases_list.append(response.choices[0].text)

    with open("cases.txt", "w") as f:
        for case in test_cases_list:
            f.write(case + '\n')
    return test_cases_list


test_cases = generate_test_cases_with_input(1)

generate_scenarios = input("Do you want to generate user stories? (Y/N): ")
if generate_scenarios.upper() == "Y":
    test_cases_string = '\n'.join(test_cases)
    scenario_prompt = f"Imagine you are a Product Owner for the mobile app, and you have been given the task of writing user stories based on following test cases: '{test_cases_string}'."
    scenario_response = openai.Completion.create(
            engine="text-davinci-003",
            temperature=0.79,
            max_tokens=3100,
            prompt=scenario_prompt
        )
    with open("stories.txt", "w") as g:
        g.write(scenario_response.choices[0].text)
else:
    print("Script Ended")
