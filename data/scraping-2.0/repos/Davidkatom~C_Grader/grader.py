import os
import subprocess
from pathlib import Path
import re

import openpyxl
from openai import OpenAI

openai_api_key = os.environ.get('OPENAI_KEY')

root_dir = "C:/Grading/ass1"  # Replace with the path to your root directory
processed_dir = "C:/Grading/ass1/ProcessedQ1"  # Replace with the path to your Processed directory
c_files = [os.path.join(processed_dir, file) for file in os.listdir(processed_dir) if file.endswith('.c')]


def grade_c_program(c_file, input_file, correct_file):
    # Compile the C program
    tests_failed = set()
    program_name = "program.exe"
    compile_command = f"gcc {c_file} -o {program_name}"
    result = subprocess.run(compile_command, shell=True, text=True, capture_output=True)

    # Check for compilation errors
    if result.returncode != 0:
        tests_failed.add(f"Compilation Error. Grade: 0")
        return 0, tests_failed

    # Read input lines from the file
    try:
        with open(input_file, 'r') as file:
            input_lines = file.readlines()
    except IOError:
        tests_failed.add("Error reading input file")
        return 0, tests_failed

    # Read correct lines from the file
    try:
        with open(correct_file, 'r') as file:
            correct_lines = file.readlines()
        correct_lines = ''.join(correct_lines)
    except IOError:
        tests_failed.add("Error reading input file")
        return 0, tests_failed

    # Initialize variables for output aggregation and grading
    aggregated_output = []
    run_errors = False

    # Run the compiled program for each line in the input file
    for input_data in input_lines:
        process = subprocess.Popen(program_name, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   shell=True, text=True)
        output, errors = process.communicate(input=input_data)

        # Check for runtime errors
        if errors:
            run_errors = True
            break

        aggregated_output.append(output)

    if run_errors:
        tests_failed.add(f"Runtime Error. Grade: 0")
        return 0, tests_failed

    correct_lines = correct_lines.strip().split('___')

    def extract_numbers(text):
        # Use regular expression to find all numbers
        return re.findall(r'-?\d+\.?\d*', text)

    for i in range(len(correct_lines)):
        desired_numbers = extract_numbers(correct_lines[i])
        student_numbers = extract_numbers(aggregated_output[i])
        if desired_numbers != student_numbers:
            tests_failed.add(f"Test {input_lines[i].strip()} failed. expected: {desired_numbers}, actual: {student_numbers} \n")

    correct_count = len(input_lines) - len(tests_failed)
    # Calculate the grade
    grade = (correct_count / len(input_lines)) * 100
    return grade, tests_failed


def grade_c_code(c_file_path, openai_api_key):
    # Read the C file
    try:
        with open(c_file_path, 'r', encoding='utf-8') as file:
            c_code = file.read()
    except IOError:
        return "Error reading C file."

    # Set up OpenAI GPT-4 API
    client = OpenAI(api_key=openai_api_key)

    # Prepare the prompt for GPT-4

    prompt = ("Analyze the following C code for coding standards and provide feedback in a structured format. "
              "Only list significant issues that notably affect the code's readability, maintainability, "
              "or functionality. Don't assume that the code neads more features or functionality.\n"
              "Use bullet points for the following categories:\n"
              "1. Comments: (Only mention if there's a significant lack of comments where necessary for understanding complex logic)\n"
              "2. Naming Conventions: (Only note issues if naming conventions lead to confusion or misunderstanding of the code purpose)\n"
              "3. Magic Numbers: (Mention only if the use of magic numbers significantly hinders understanding of what the code is doing)\n"
              "4. Tabs for Indentation: (Only raise an issue if improper indentation severely affects the readability of the code)\n"
              "5. Code Efficiency and Simplicity: (Only comment on efficiency and simplicity if there are glaring inefficiencies or overly complex structures that make the code hard to understand)\n\n"
              "After the list, provide a grade on a scale of 1 to 5 based on these criteria. "
              "A score of 5 means the code has no significant issues, and a lower score indicates notable problems in the listed categories.\n\n"
              f"C Code:\n\n{c_code}\n\n"
              "Please format your response as follows:\n"
              "Comments:\n- [Significant Issue 1]\n- [Significant Issue 2]\n...\n"
              "Naming Conventions:\n- [Significant Issue 1]\n- [Significant Issue 2]\n...\n"
              "Magic Numbers:\n- [Significant Issue 1]\n- [Significant Issue 2]\n...\n"
              "Tabs for Indentation:\n- [Significant Issue 1]\n- [Significant Issue 2]\n...\n"
              "Code Efficiency and Simplicity:\n- [Significant Issue 1]\n- [Significant Issue 2]\n...\n"
              "Grade = [only a number on a scale of 1 to 5] for example Grade = 3 ")

    # Send the prompt to the GPT-4 API
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=1.5,
            model="gpt-3.5-turbo",
        )
        report = chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error in API call: {e}"

    # Output the report and grade
    return report


def extract_and_print_grade(report):
    # Split the report based on the known phrase
    parts = report.split("Grade = ")
    if len(parts) > 1:
        # Further split to isolate the grade
        grade_part = parts[1].split()[0]  # Assumes grade is followed by a space or end of string
    else:
        print("Grade not found in the report.")
        grade_part = -1
    return grade_part


def grade_and_populate_excel(c_files, input_text, correct_output, openai_api_key):
    # Initialize an Excel workbook and create a new sheet
    wb = openpyxl.Workbook()
    sheet = wb.active

    # Write headers to the Excel sheet
    sheet['A1'] = 'C File'
    sheet['B1'] = 'Code Grade'
    sheet['C1'] = 'Style Grade'
    sheet['D1'] = 'Report'
    sheet['E1'] = 'Total Grade'

    # Directory where the reports and code files are located
    reports_directory = "reports"

    # Loop through the C files
    for index, c_file in enumerate(c_files, start=2):  # Start from row 2 for data
        print("report for", c_file, ":")
        # Grade the code
        code_grade, tests_failed = grade_c_program(c_file, input_text, correct_output)
        print("code grade =", code_grade)

        # # OpenAI API call and write to a file
        with open(os.path.join(reports_directory, Path(f'{c_file}.txt').name), "w") as file:
            report = grade_c_code(c_file, openai_api_key)
            file.write(f"Report for {c_file}:\n{report}\n\n")

        # Extract and print the style grade
        style_grade = extract_and_print_grade(report)
        print("style grade =", style_grade)
        # style_grade = 5  # CHANGE THIS TO THE GRADE
        # Calculate the total grade
        total_grade = 0.7 * float(code_grade) + 0.3 * float(style_grade) * 20
        print("Total grade =", total_grade)
        #report = "Tests failed:\n" + ''.join(tests_failed)

        report = report + "\n" + "Tests failed:\n" + ''.join(tests_failed)
        # Populate the Excel sheet with the grades and report
        sheet[f'A{index}'] = c_file
        sheet[f'B{index}'] = code_grade
        sheet[f'C{index}'] = style_grade
        sheet[f'D{index}'] = report
        sheet[f'E{index}'] = total_grade

    # Save the Excel workbook
    wb.save(root_dir + "/grading_results.xlsx")


# Example usage


input_text = root_dir + "/inputQ1.txt"
correct_output = root_dir + "/outputQ1.txt"

grade_and_populate_excel(c_files, input_text, correct_output, openai_api_key)
