import json
import openai

def load_api_details():
    with open("api_details.json", "r") as file:
        api_details = json.load(file)
    return api_details

def get_gpt_response(prompt, api_details):
    openai.api_key = api_details["api_key"]
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def prompt_input(prompt):
    print(prompt)
    lines = []
    while True:
        line = input()
        if line == "done":
            break
        elif line == "gpt":
            # Generate GPT response based on known information
            api_details = load_api_details()
            chat_history = '\n'.join(lines)
            gpt_prompt = "{}\nYou:".format(chat_history)
            gpt_response = get_gpt_response(gpt_prompt, api_details)
            lines.append("Chat-GPT: {}".format(gpt_response))
            print("Chat-GPT: {}".format(gpt_response))
        else:
            lines.append(line)
    return lines

def print_report(brief, hypothesis, evidence, assumptions, alternate_explanations):
    print("\n--- Report ---")
    print("Brief: {}".format(brief))
    print("\nHypothesis: {}".format(hypothesis))
    print("\nEvidenced by:")
    for line in evidence:
        print("- {}".format(line))
    print("\nAssumptions:")
    for line in assumptions:
        print("- {}".format(line))
    print("\nAlternate Explanations:")
    for line in alternate_explanations:
        print("- {}".format(line))

def main():
    # Describe the current situation
    current_situation = prompt_input("Describe the current situation?")
    
    # What is your current leading hypothesis?
    leading_hypothesis = prompt_input("What is your current leading hypothesis?")
    
    # What assumptions are you making?
    assumptions = prompt_input("What assumptions are you making?")
    
    # Are there any alternative explanations for this?
    alternative_explanations = prompt_input("Are there any alternative explanations for this?")
    
    # Based on this, now list all possible hypotheses.
    possible_hypotheses = prompt_input("Based on this, now list all possible hypotheses.")
    
    # Input the evidence that supports the following hypotheses.
    hypotheses_evidence = {}
    for hypothesis in possible_hypotheses:
        evidence = prompt_input("Input the evidence that supports the hypothesis: '{}'".format(hypothesis))
        hypotheses_evidence[hypothesis] = evidence
    
    # Which hypothesis is currently most supported by the evidence?
    print("\n--- Select Leading Hypothesis ---")
    for i, hypothesis in enumerate(possible_hypotheses):
        print("{}. {}".format(i+1, hypothesis))
        print("   Evidence:")
        for line in hypotheses_evidence[hypothesis]:
            print("   - {}".format(line))
    
    selection = input("Enter the number of the most supported hypothesis: ")
    selected_hypothesis = possible_hypotheses[int(selection)-1]
    
    # Print report
    print_report('\n'.join(current_situation), selected_hypothesis, hypotheses_evidence[selected_hypothesis], assumptions, alternative_explanations)

    # Ask if user wants to print the report
    print_report_option = input("\nPrint report? (Y/N): ")
    if print_report_option.upper() == 'Y':
        print_report('\n'.join(current_situation), selected_hypothesis, hypotheses_evidence[selected_hypothesis], assumptions, alternative_explanations)
    else:
        print("Report not printed.")

if __name__ == "__main__":
    main()
