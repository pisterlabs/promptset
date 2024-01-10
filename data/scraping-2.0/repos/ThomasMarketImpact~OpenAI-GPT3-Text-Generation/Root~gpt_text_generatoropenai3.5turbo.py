import os
import openai
from dotenv import load_dotenv
import datetime
import time
import logging
import glob

load_dotenv('Root/.env.local')
openai.api_key = os.environ['OPENAI_API_KEY']

MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.2

CONFIG = {
    'system1_file_path': 'Root/data/Promptfiles/system1.txt',
    'system2_file_path': 'Root/data/Promptfiles/system2.txt',
    'user_file_path': 'Root/data/Promptfiles/user.txt',
    'output_dir': 'Root\output',
}


logger = logging.getLogger(__name__)

def validate_input_files(*file_paths):
    """Check if the input files exist."""   
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

def read_file(file_path):
    """Read the contents of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def prepare_input_text(system1_content, system2_content, user_file_content):
    """Prepare the input text for the GPT-3 API."""
    roles = ["system", "system", "user"]
    contents = [system1_content, system2_content, user_file_content]
    return [{"role": r, "content": c} for r, c in zip(roles, contents)]

def generate_text(input_text):
    """Generate text using the GPT-3 API."""
    print("generating text... may take a few momnents")
    response = openai.ChatCompletion.create(
        model=MODEL_NAME, 
        messages=input_text,
        n=1,
        stop=None,
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content.strip()

def save_output_file(generated_text, user_file_path):
    """Save the generated text to an output file."""
    file_name = os.path.splitext(os.path.basename(user_file_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(CONFIG['output_dir'], f"{file_name}_output_{timestamp}.txt")

    with open(output_file_path, 'w') as output_file:
        output_file.write(generated_text)

    return output_file_path


def generate_text_file(user_file_path, system1_file_path, system2_file_path):
    """Generate text using the GPT-3 API and save it to an output file."""
    try:
        validate_input_files(user_file_path, system1_file_path, system2_file_path)

        system1_content = read_file(system1_file_path)
        system2_content = read_file(system2_file_path)
        user_file_content = read_file(user_file_path)

        input_text = prepare_input_text(system1_content, system2_content, user_file_content)

        generated_text = generate_text(input_text)

        output_file_path = save_output_file(generated_text, user_file_path)

        logger.info(f"Generated text:\n{generated_text}")
        logger.info(f"Output file saved at: {output_file_path}")

        return input_text

    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
        return None

    except openai.error.RateLimitedError as e:
        logger.warning(f"API rate limit exceeded. Pausing for {e.retry_after} seconds.")
        time.sleep(e.retry_after)
        return None

    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        return None

def main():
    """Main function that runs the script."""
    logging.basicConfig(level=logging.INFO)

    input_text = generate_text_file(
        CONFIG['user_file_path'], 
        CONFIG['system1_file_path'], 
        CONFIG['system2_file_path']
    )

    if input_text is not None:
        output_file_path = os.path.join(CONFIG['output_dir'], f"{os.path.splitext(CONFIG['user_file_path'])[0]}_output_*.txt")
        output_files = glob.glob(output_file_path)
        if len(output_files) > 0:
            with open(output_files[-1], 'r') as output_file:
                generated_text = output_file.read()
                print(f"Generated text:\n{generated_text}")
        else:
            print("Output file not found.")
        
if __name__ == '__main__':
    main()
