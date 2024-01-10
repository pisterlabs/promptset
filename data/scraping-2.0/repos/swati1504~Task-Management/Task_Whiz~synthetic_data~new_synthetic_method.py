import openai
import pandas as pd

# Replace with your OpenAI API key
OPENAI_API_KEY = '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'

# Setup OpenAI API client
openai.api_key = OPENAI_API_KEY

reference_df = pd.read_csv("reference_data.csv")

rules = [
    "Age should be between 22 to 65. Age is positively correlated with marriage status, with employees over 30 usually married.",
    "Backend roles typically have older employees with higher salaries and senior positions. Frontend roles generally have younger employees with decent salaries and a mix of junior and senior positions.",
    "HR roles are predominantly held by females. Older employees tend to have more absences and higher salaries.",
    "Employee's salary is partly determined by their position, rating, projects completed, and monthly hours. Morale is positively influenced by salary.",
    "Stress and Burnout Score is positively affected by project workload and difficulty, and negatively by deadlines met, salary, and absences.",
    "Employee data may contain variability and stochastic behavior to account for hidden factors. Over time, features such as salary may change for the same employee."
]

# Function to generate synthetic data using OpenAI
def generate_synthetic_data(reference_df, rules, num_examples=10):
    # Convert reference dataframe to a string representation
    data_examples = reference_df.to_string(index=False)

    # Create the prompt for the API
    prompt = f"Given the small dataset:\n\n{data_examples}\n\nand the following rules:\n{rules}\n\nGenerate {num_examples} new synthetic data entries following the same pattern and rules."

    # Call OpenAI API to generate data
    response = openai.completions.create(
        model="text-davinci-003",  # Use the latest available model
        prompt=prompt,
        max_tokens=1024,
        n=num_examples
    )

    # Extract generated text and convert into a list of dictionaries
    generated_text = response.choices[0].text.strip()
    synthetic_data_entries = [line for line in generated_text.split('\n') if line]

    # Parse generated data into a pandas DataFrame
    synthetic_data_list = []
    for entry in synthetic_data_entries:
        # Assuming the output is in a comma-separated values format or similar
        # This will need adjustment based on your specific formatting
        entry_data = entry.strip().split(',')
        synthetic_data_list.append(entry_data)
    
    synthetic_df = pd.DataFrame(synthetic_data_list, columns=reference_df.columns)
    return synthetic_df

dataset = generate_synthetic_data(reference_df, rules, 1000)
dataset.to_csv('synthetic_data.csv', index=False)