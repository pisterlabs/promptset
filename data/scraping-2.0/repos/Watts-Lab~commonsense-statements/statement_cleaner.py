import os
import pandas as pd
import openai

# Set up the paths for the inputs and outputs directories
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"


openai.api_key = os.environ["OPENAI_API_KEY"]


def process_line(line):
    """Process a line of text using the OpenAI API."""
    print("Processing line:", line)
    return "Processed " + line  # Placeholder for the actual API response


os.makedirs(OUTPUTS_DIR, exist_ok=True)

input_files = [f for f in os.listdir(INPUTS_DIR) if f.endswith(".csv")]

for input_file in input_files:
    input_path = os.path.join(INPUTS_DIR, input_file)
    output_file = f"{os.path.splitext(input_file)[0]}_out.csv"
    output_path = os.path.join(OUTPUTS_DIR, output_file)

    # Read the input CSV into a DataFrame
    input_df = pd.read_csv(input_path)

    if "statements" not in input_df.columns:
        print(f"'statements' column not found in {input_file}. Skipping this file.")
        continue

    # Initialize the output DataFrame
    if os.path.exists(output_path):
        output_df = pd.read_csv(output_path)
        # Ensure 'statements' column exists in output_df
        if "statements" not in output_df.columns:
            print(
                f"'statements' column not found in {output_file}. Creating a new file."
            )
            output_df = pd.DataFrame(columns=["statements", "processed"])
    else:
        output_df = pd.DataFrame(columns=["statements", "processed"])

    # Get new statements
    new_statements_mask = ~input_df["statements"].isin(output_df["statements"])
    new_statements = input_df.loc[new_statements_mask, "statements"]

    # Process new statements
    if not new_statements.empty:
        processed_data = [
            {"statements": line, "processed": process_line(line)}
            for line in new_statements
        ]

        new_processed_df = pd.DataFrame(processed_data)

        output_df = pd.concat([output_df, new_processed_df], ignore_index=True)

        output_df.to_csv(output_path, index=False)
        print(f"Processed new data appended to {output_path}")
    else:
        print(f"No new statements to process in {input_file}. Skipping processing.")

print("Processing completed.")
