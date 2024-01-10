import openai
from parser_helper import parse_nmap_xml

def query_openai(prompt):
    # Query the OpenAI API with the prompt and return the response.
    # This fucntion is called within the run_analysis function below.
    try:
        response = openai.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=250
        )
        return response.choices[0].text.strip()
    except openai.OpenAIError as e:
        print(f"An error occurred while querying the OpenAI API: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit()

def run_analysis(api_key, xml_file_path):
    # This is the main function that runs the analysis based on the Nmap XML file.
    # This function is called from the main function in setup.py.\

    openai.api_key = api_key # Set the API key

    nmap_data = parse_nmap_xml(xml_file_path) # Parse the Nmap XML file
    if not nmap_data.strip():
        print("No data was found in the Nmap XML file. Exiting the program.")
        exit()

    # Define pre-prompt text.
    pre_prompt = "You are a cybersecurity expert providing advice on a penetration test based on nmap data. Provide advice on how to proceed next in the test. In your response, ensure you summarise how many hosts, ports, and services were found. Then proceed to provide your recommendations, categorising them by each host. For example, after providing a summary, you could say: 'For host 1, I recommend...'"

    # Create the prompt for OpenAI
    full_prompt = f"{pre_prompt}\n\nNetwork scan report:\n{nmap_data}\n\nRecommendations:"
    print("The data from the provided Nmap scan has been passed to OpenAI. Please wait for recommendations...\n\n")

    try:
    # Query OpenAI 
        response = query_openai(full_prompt)
        print(response)

        # Check if the response suggests further scanning
        response_phrases = [
            "further scanning", 
            "additional scans", 
            "running", 
            "next steps", 
            # Add more phrases that might indicate the response suggested subsequent scanning.
        ]

        # Check if any of the phrases are in the response
        if any(phrase in response for phrase in response_phrases):
            while True:  # Start a loop that will continue until broken
                user_input = input("\nDo you need help crafting further scan commands? (yes/no): \n").strip().lower()

                if user_input in ["yes", "no"]:  # Check if input is valid
                    break  # Exit the loop if input is valid
                else:
                    print("Invalid input. Please type 'yes' or 'no'.\n") 
                        
            if user_input == "yes":
                # Create a new prompt asking for specific Nmap command based on the recommendation
                new_prompt = f"Based on the previous recommendation for further scanning, what specific Nmap command should be used? Context: {nmap_data}\n"
                # Query OpenAI for the Nmap command
                nmap_command_response = query_openai(new_prompt)
                print("Suggested Nmap Command:\n")
                print(nmap_command_response)

        # After handling 'yes' or 'no', ask if the user has any other questions
        while True:  # Loop for handling other cybersecurity questions
            other_query = input("\nDo you have any other cybersecurity questions? (yes/no): \n").strip().lower()

            if other_query == "yes":
                user_question = input("\nFollow-up question:\n")
                response = query_openai(user_question)
                print("\nResponse to your question:\n")
                print(response)
                        
            elif other_query == "no":
                print("No further questions. Exiting the program. Have a great day!")
                break  # Break out of the loop to exit
                exit()
                
            else:
                print("Invalid input. Please type 'yes' or 'no'.\n")  # Handle invalid input

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting the program. Have a great day!")
        exit()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit()