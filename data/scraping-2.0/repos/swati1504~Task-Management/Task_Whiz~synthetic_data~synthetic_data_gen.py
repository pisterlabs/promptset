import openai
import csv

# Replace 'your_api_key_here' with your actual key from OpenAI.
openai.api_key = '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'

# Read reference examples from a CSV file
def read_reference_examples(csv_file_path):
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        reference_examples = list(reader)
    # Assume the first row is the header, so we skip it
    return reference_examples[1:]  

reference_examples = read_reference_examples('reference_data.csv')

# Proceed with the rest of the code as before
# ...

def create_prompt(reference_examples, rules):
    example_str = "\n".join(", ".join(example) for example in reference_examples)
    rules_description = "\n".join(rules)
    # ...
    prompt = (
        f"Given the following reference examples of employee records, generate one new synthetic record that follows the provided guidelines. Use a similar format as the examples.\n\n"
        f"Reference examples:\n"
        f"(Employee Name, ID, Gender, Age, Married, Role, Salary, Position, Absences, "
        f"Projects_Completed, Mean Monthly Hours, Years in the company, Joining_Year, "
        f"Current_Employ_Rating, Moral, Stress & Burnout Score, Ongoing_Project_Count, Projects_Within_Deadline, "
        f"Project_Start_Date, Project_Description, Project_Difficulty, Project_Deadline, Manager_ID)\n\n"
        f"{example_str}\n"
        f"Please generate a new employee record following a similar pattern and the rules:\n{rules_description}\n"
    )
    return prompt

# Ensure that you have a file named 'reference_data.csv' and the rest of the code is correct...
# Function to call the GPT-3 API and create a synthetic entry
def generate_synthetic_entry(prompt):
    response = openai.completions.create(
        model="text-davinci-003",  # or another model version
        prompt=prompt,
        max_tokens=200  # adjust as needed
    )
    data = response.choices[0].text.strip()
    return data.split(", ")

# Creating the CSV file with synthetic data
csv_header = [
    "Employee Name", "ID", "Gender","Age", "Married", "Role", "Salary", "Position", "Absences", "Projects_Completed", "Mean Monthly Hours", "Years in the company", "Joining_Year", "Current_Employ_Rating", 
    "Moral", "Stress & Burnout Score", "Ongoing_Project_Count", "Projects_Within_Deadline",
    "Project_Start_Date", "Project_Description", "Project_Difficulty", "Project_Deadline", "Manager_ID"
]

# Adjust the number of synthetic entries you desire
number_of_entries = 2000

# Rules that you want to enforce - add your own as required
rules = [
    "Generate samples of the same 50 unique employees. Each sample is an entry at a particular time instace",
    "Age should be between 22 to 65. Age is positively correlated with marriage status, with employees over 30 usually married.",
    "Backend roles typically have older employees with higher salaries and senior positions. Frontend roles generally have younger employees with decent salaries and a mix of junior and senior positions.",
    "HR roles are predominantly held by females. Older employees tend to have more absences and higher salaries.",
    "Employee's salary is partly determined by their position, rating, projects completed, and monthly hours. Morale is positively influenced by salary.",
    "Stress and Burnout Score is positively affected by project workload and difficulty, and negatively by deadlines met, salary, and absences.",
    "Employee data may contain variability and stochastic behavior to account for hidden factors. Over time, features such as salary may change for the same employee."
]

with open('synthetic_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

    for _ in range(number_of_entries):
        prompt = create_prompt(reference_examples, rules)
        synthetic_entry = generate_synthetic_entry(prompt)
        writer.writerow(synthetic_entry)
        print(f"Generated synthetic entry: {synthetic_entry}")

print("Finished generating synthetic data.")