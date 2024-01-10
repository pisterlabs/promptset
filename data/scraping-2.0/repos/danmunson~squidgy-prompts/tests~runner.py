
import openai
import os
import pytest
import yaml
import difflib

class Gpt3Client:

    def __init__(self):
        openai.api_key = os.environ['OPENAI_API_KEY']

    def invoke_from_file(
        self, 
        prompt_file: str, 
        params: dict[str,str],
        additional_text: str = None,
        engine='text-davinci-003',
        temperature=0, 
        stop=['}', '###'], 
        max_tokens=300,
        frequency_penalty=0.15,
        presence_penalty=0.3,
    ) -> str:
        with open(prompt_file) as f:
            prompt = f.read()

            if additional_text is not None:
                prompt += additional_text

            for param in params:
                if params[param] is not None:
                    prompt = prompt.replace("{%s}" % param, params[param])

            return self.invoke(prompt, engine, temperature, stop, max_tokens, frequency_penalty, presence_penalty)

    def invoke(
        self, 
        prompt: str, 
        engine='text-davinci-003',
        temperature=0, 
        stop=['}', '###'],
        max_tokens=300,
        frequency_penalty=0.15,
        presence_penalty=0.3,
    ) -> str:
        response = openai.Completion.create(
            prompt=prompt,
            model=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=1,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            timeout=15
        )

        return response['choices'][0]['text'].strip()

@pytest.fixture
def gpt3_client():
    return Gpt3Client()

def load_tests(file_name: str) -> list[dict]:
    if file_name is not None:
        with open(file_name) as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)

            return { file_name : doc }

    # lists all yaml files in the tests directory and parses them into Python objects
    tests = {}
    for root, dirs, files in os.walk("."):
        for file in files:
            print(file)
            if file.endswith(".yaml"):
                with open(os.path.join(root, file)) as f:
                    doc = yaml.load(f, Loader=yaml.FullLoader)

                    tests[file] = doc

    return tests

def run_tests(file_name: str, test_name_to_run: str):
    tests = load_tests(file_name)
    gpt3_client = Gpt3Client()
    results = {}

    print(f"Running tests...")
    for file in tests:
        file_name = os.path.split(file)[-1]
        test_suite = tests[file]
        print("Running " + file_name)
        
        for test_name in test_suite:
            if test_name_to_run is not None and test_name != test_name_to_run:
                continue

            test_definition = test_suite[test_name]

            if 'base_prompt' not in test_definition:
                raise Exception(f"Test {test_name} must contain a base prompt")

            base_prompt = test_definition['base_prompt']
            
            prompt = None
            if 'prompt' in test_definition:
                prompt = test_definition['prompt']
                if prompt is not None:
                    prompt + "\n"

            params = {}
            if 'params' in test_definition:
                params = test_definition['params']

            stop = stop=['bob:', 'Bob:']
            if 'stop' in test_definition:
                stop = test_definition['stop']

            expected = test_definition['expected']
            expected = expected.strip()

            reply = gpt3_client.invoke_from_file(base_prompt, params, prompt, stop=stop)

            success = reply == expected

            if success:
                print(f"{test_name}: Passed")
            else:
                print(f"{test_name}: Failed")

                # create a character diff of the expected vs reply strings:
                diff = difflib.ndiff(expected.splitlines(keepends=True), reply.splitlines(keepends=True))
                print(''.join(diff))

                print()    

            results[test_name] = {
                'prompt' : prompt,
                'expected': expected,
                'reply' : reply,
                'success' : success
            }