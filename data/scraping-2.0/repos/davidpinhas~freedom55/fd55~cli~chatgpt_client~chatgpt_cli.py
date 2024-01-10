import time
import re
import openai
import shutil
import logging
import itertools
from fd55.utils.fd55_config import Config
config = Config()


class ChatGPT:
    def __init__(self):
        openai.api_key = f"{str(config.get('AI', 'api_key'))}"
        self.engine = "text-davinci-003"

    def backup_modified_file(file):
        """ Create a backup for the modified file """
        shutil.copy2(file, file + '.bak')
        logging.info(f"Created backup of '{file}' as '{file}.bak'")

    def remove_first_comment(file):
        """ Remove all comments at the head of the file """
        with open(file, 'r') as f:
            lines = f.readlines()
        with open(file, 'w') as f:
            found_valid_line = False
            for line in lines:
                if not found_valid_line:
                    if re.search(
                            r"[a-zA-Z0-9À-ÿ]+",
                            line) and not (
                            line.startswith(" ") or line.startswith("#") and not line.startswith("#!")):
                        found_valid_line = True
                        f.write(line)
                else:
                    f.write(line)

    def iterate_code_rewrite(self, file=None, iterations=0, prompt=None):
        """ Iterate over a file with the same prompt """
        if iterations > 0 and file is not None:
            retry_count = 0
            logging.info(f"Backing up file '{file}'")
            ChatGPT.backup_modified_file(file)
            ChatGPT.remove_first_comment(file=file)
            logging.info(f"Running {iterations} iterations on file '{file}'")
            for i in range(iterations):
                with open(file, 'r') as f:
                    file_content = f.read()
                    new_prompt = ""
                    new_prompt += prompt + '. File to modify: \n' + \
                        file_content + "\n\n Return the improved version in full."
                    logging.debug(f"Manifested new prompt:\n{prompt}")
                while True:
                    output = openai.Completion.create(
                        engine=self.engine,
                        prompt=new_prompt,
                        max_tokens=2048,
                        temperature=0,
                        stream=False
                    )
                    if not output.choices[0].text:
                        retry_count += 1
                        logging.warning(
                            f"Received empty output from OpenAI, retrying...")
                        if retry_count == 10:
                            logging.error(
                                "Failed to get a response from OpenAI after 10 retries")
                        time.sleep(5)
                        pass
                    break
                with open(file, 'w') as f:
                    f.write(output.choices[0].text)
                logging.info(f"Iteration {i+1} - Finished updating file")
            logging.info(f"All iterations on file '{file}' finished")

    def send_openai_request(
            self,
            prompt,
            full_output=False,
            file=None,
            iterations=None):
        """ Send OpenAI request """
        try:
            logging.info("Creating request to OpenAI")
            if file is not None:
                logging.info(f"Manifesting prompt with file '{file}'")
                ChatGPT.remove_first_comment(file)
                with open(file, 'r') as f:
                    file_content = f.read()
                    original_prompt = prompt
                    prompt += prompt + '. File to improve: \n' + file_content + \
                        "\n\n Return the improved version in full."
                    logging.debug(f"Manifested new prompt:\n{prompt}")
                if iterations:
                    iterations = int(iterations)
                    if int(iterations) and iterations > 0:
                        ChatGPT().iterate_code_rewrite(
                            file=file, iterations=iterations, prompt=original_prompt)
                        exit()
            if file is None and iterations:
                logging.error(f"Iterations argument requires a '--file'")
                exit()
            stream = False if full_output else True
            output = openai.Completion.create(
                engine=self.engine,
                prompt=prompt,
                max_tokens=2048,
                temperature=0,
                stream=stream
            )
            if full_output is not False:
                logging.info("Waiting for full output generation")
                full_output = output.choices[0].text
                print(full_output)
                logging.info("Finished generation")
            else:
                logging.info("Recieved answer:")
                print("#################################")
                print("######## OpenAI Response ########")
                print("#################################\n")
                for event in itertools.islice(output, 1, None):
                    event_text = event['choices'][0]['text']
                    print(f"{event_text}", end="", flush=True)
                print("\n\n#################################")
        except Exception as e:
            logging.error(f"Failed to get a response from OpenAI, Reason: {e}")
            exit()
