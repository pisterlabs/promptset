import os
import openai

# Set your OpenAI API key
import config   
openai.api_key = config.api_key

# Function to generate documentation comments using ChatGPT
def generate_documentation(code):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are generating code documentation for a project written in the Julia programming language. The documentation should at least include all the functions as well as their input and output. It should be written in Markdown syntax. The project is a mechanistic species distribution model using stochastic and metabolistic processes."},
            {"role": "user", "content": code}
        ]
    )
    return response['choices'][0]['message']['content']

# Function to process a Python script and generate documentation
def process_script(script_path, output_directory):
    try:
        print(f"Processing {script_path}")
        with open(script_path, "r") as file:
            code = file.read()
            
            # Generate documentation comments
            documentation = generate_documentation(code)
            
            # Create a file for documentation in the output directory
            script_name = os.path.basename(script_path)
            output_file_name = os.path.splitext(script_name)[0] + ".md"
            output_file_path = os.path.join(output_directory, output_file_name)
            
            with open(output_file_path, "w") as doc_file:
                doc_file.write(documentation)
                
            print(f"Documentation saved to {output_file_path}")
    
    except Exception as e:
        print(f"An error occurred processing {script_path}: {str(e)}")

# Function to process all Python scripts in a directory
def process_directory(directory_path, output_directory):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".jl"):
                script_path = os.path.join(root, file)
                process_script(script_path, output_directory)

def main():
    input_directory = "../MetaRange/src"
    output_directory = "Results"
    process_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()
