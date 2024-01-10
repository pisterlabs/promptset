from modules import hypothesis_generation, verification_execution
from langchain.llms import OpenAI
import datetime
import warnings
import os

warnings.simplefilter('ignore')

def save_outputs(outputs, now):
    base_path_to_save_outputs = f'outputs/{now}'
    os.makedirs(base_path_to_save_outputs, exist_ok=True)
    for key, value in outputs.items():
        if key == 'verification_code' or key == 'package_install_code' or key == 'verification_code_updated':
            extension = 'py'
        else:
            extension = 'txt'
        with open(f'{base_path_to_save_outputs}/{key}.{extension}', 'w') as f:
            f.write(value)

def main():
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    llm = OpenAI(model_name="gpt-4", temperature=0.0)

    with open('input_data/problem.txt') as f:
        problem = f.read()
        print(problem)

    outputs = {}

    # Hypothesis Generation
    hypothesis_generator = hypothesis_generation.HypothesisGenerator(llm)
    hypothesis = hypothesis_generator(problem)['hypothesis']
    outputs.update(hypothesis_generator.outputs)
    del hypothesis_generator

    # Verification Execution
    verification_executor = verification_execution.VerificationExecutor(llm)
    verification_executor(problem, hypothesis)
    outputs.update(verification_executor.outputs)
    del verification_executor

    # Save outputs
    save_outputs(outputs, now)

if __name__ == "__main__":
    main()