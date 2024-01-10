"""
This script generates explanations for the outputs of a network intrusion detection system (NIDS).
It uses the GPT-3 API to generate explanations for the outputs.
It then extracts relevant information from the GPT-3 response and formats it as a detailed report.
Created by Andres Haro, 2023. 

"""

import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate explanation for a given output
def generate_explanation(output_text):
    prompt = f"Explain the following output:\n{output_text}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=800  # Adjust as needed
    )
    return response.choices[0].text.strip()

# Function to extract information from the GPT-3 response
def extract_information(gpt_response):
    # Implement your logic to extract relevant information
    # Example: You might use regex or keyword-based extraction
    source_ip = "123.456.789.0"
    destination_ip = "987.654.321.0"
    host = "example.com"
    vulnerability_type = "SQL Injection"
    description = "A vulnerability allowing unauthorized SQL queries."
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    request_type = "GET"
    is_successful = True  # Set based on your logic

    return {
        "Source IP": source_ip,
        "Destination IP": destination_ip,
        "Host": host,
        "Type of vulnerability": vulnerability_type,
        "Description of the problem": description,
        "User-Agent": user_agent,
        "Request Type": request_type,
        "Is Successful": is_successful,
    }

# Function to dynamically generate solutions based on the vulnerability
def generate_solutions(vulnerability_type):
    if vulnerability_type == "SQL Injection":
        return [
            "Apply input validation for user inputs.",
            "Use parameterized queries to prevent SQL injection.",
            "Regularly update and patch database software.",
        ]
    elif vulnerability_type == "XSS":
        return [
            "Encode user input before rendering it in HTML.",
            "Implement Content Security Policy (CSP) headers.",
            "Use secure coding practices to validate and sanitize input.",
        ]
    # Add more cases for other vulnerability types as needed
    else:
        return ["Implement best practices for securing web applications."]

# Function to format the information as a detailed report
def format_report(information):
    report = f"Report:\n\n"
    report += f"Source IP: {information['Source IP']}\n"
    report += f"Destination IP: {information['Destination IP']}\n"
    report += f"Host: {information['Host']}\n"
    report += f"Type of vulnerability: {information['Type of vulnerability']}\n"
    report += f"Description of the problem: {information['Description of the problem']}\n"
    report += "\nPossible Solutions:\n"
    # Dynamically generate solutions based on the vulnerability
    solutions = generate_solutions(information['Type of vulnerability'])
    for solution in solutions:
        report += f"- {solution}\n"
    report += f"\nUser-Agent: {information['User-Agent']}\n"
    report += f"Request Type: {information['Request Type']}\n"
    report += f"Is Successful: {information['Is Successful']}\n"

    return report

# Function to process and generate explanations for all files in a folder
def process_folder(input_folder, output_folder):
    # List all files in the input folder
    input_files = os.listdir(input_folder)

    # Process each file
    for input_file in input_files:
        input_file_path = os.path.join(input_folder, input_file)
        output_file_path = os.path.join(output_folder, input_file)

        # Read the content of the input file
        with open(input_file_path, 'r') as file:
            input_text = file.read()

        # Generate explanation using GPT-3
        explanation = generate_explanation(input_text)

        # Extract information from the GPT-3 response
        information = extract_information(explanation)

        # Format the information as a detailed report
        report = format_report(information)

        # Save the report to the output file
        with open(output_file_path, 'w') as file:
            file.write(report)

# Input and output folder paths
input_folder_path = "[your_input_folder_path]"
output_folder_path = "[your_output_folder_path]"

# Process the folder and generate explanations
process_folder(input_folder_path, output_folder_path)

print("Reports generated and saved to Better_Outputs folder.")
