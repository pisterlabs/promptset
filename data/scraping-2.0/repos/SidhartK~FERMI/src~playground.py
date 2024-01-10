import argparse
from typing import Optional
from dotenv import load_dotenv
import os
from tqdm import tqdm
import numpy as np

# torch
import torch

# init hugging face
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, LLMMathChain

from utils import compile_fp, convert_units, accuracy_metric
import json

load_dotenv()

INPUT_PROMPT = PromptTemplate(
                        input_variables=["context", "question"],
                        template="""{context}=QUESTION:={question}"""
                    )

OUTPUT_PROMPT = PromptTemplate(
                        input_variables=["answer", "question"],
                        template="""{answer}={program}"""
                    )

EXAMPLE_PROMPT = PromptTemplate.from_template(
                        """Input: {input}\n\n{output}"""
                    )

SYSTEM_PROMPT_STRING = """
You are a helpful assistant tasked with answering estimation questions using provided information.
You are provided with contextual information which contains facts that is relevant to answering the estimation question.
Please provide your answer in the format shown below.
"""

FEW_SHOT_PROMPT_STRING = """
Input: CONTEXT:=F1: The volume of the penny is 0.93 in**3=QUESTION:=If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be?

```python\n# Q0: If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be?\n# Q1: What is the volume of a single penny?\nA1=0.93 # volume of a single penny (in**3)\n# Q2: What is the surface area of a hollow sphere 1 atom thick which is constructed from a melted penny?\nA2=0.93 # surface area of a hollow 1 atom thick sphere constructed from a melted penny (in**2)\nA0=(4/3)*np.pi*(A2/(4*np.pi))**(3/2) # volume of a hollow 1 atom thick sphere constructed from a melted penny (in**3)```

Input: CONTEXT:=F1: It takes around 5 seconds to pluck a single strand of hair.=F2: The entire human body has 5e+6 hair follicles.=QUESTION:=How long would it take to pluck each hair out of your body one at a time?

```python\n# Q0: How long would it take to pluck each hair out of your body one at a time?\n# Q1: How long does it take to pluck a single strand of hair?\nA1=5 # time to pluck single strand of hair (s)\n# Q2: How many hairs do we have on our body?\nA2=5e+6 # number of hairs on body (hairs)\nA0=A1*A2 # time to pluck all hairs on body (s)```

Input: CONTEXT:=F1: The volume of the lecture hall is 24000 metre cube=F2: The volume of a single air molecule is 3e-24 cubic meter=QUESTION:=How many air molecules are in this lecture hall?

```python\n# Q0: How many air molecules are in this lecture hall?\n# Q1: What is the volume of the lecture hall?\nA1=24000 # volume of the lecture hall (m**3)\n# Q2: What is the volume of a single air molecule?\nA2=3e-24 # volume of single air molecule (m**3)\nA0=A1/A2 # number of air molecules in lecture hall (molecules)```

Input: CONTEXT:=F1: The average american lifespan is 78 years.=F2: Cancer cuts down the average lifespan of a person by 5 years.=QUESTION:=A cure for all cancers would increase the average American lifespan by how many years?

```python\n# Q0: A cure for all cancers would increase the average American lifespan by how many years?\n# Q1: What is the average american lifespan?\nA1=78 # average american lifespan (years)\n# Q2: Cancer cuts down the average lifespan by how many years?\nA2=5 # amount cancer reduces average lifespan by (years)\nA0=A1+A2 # average lifespan with cancer (years)```

Input: CONTEXT:=F1: The mass of a queen size mattress is 140 pounds=F2: The mass of the sun is 4.3e+30 pounds.=F3: The ratio of mass of Betelgeuse to mass of sun is 18=QUESTION:=What is the number of queen-size mattresses it would take to fill the star Betelgeuse which has 18 times the mass of the Sun?

```python\n# Q0: What is the number of queen-size mattresses it would take to fill the star Betelgeuse which has 18 times the mass of the Sun?\n# Q1: What is the mass of a queen size mattress?\nA1=140 # mass of queen size mattress (pounds)\n# Q2: What is the mass of Betelgeuse?\n# Q3: What is the mass of the sun?\nA3=4.3e+30 # mass of the sun (pounds)\n# Q4: What is the ratio of the mass of Betelgeuse to the mass of the sun?\nA4=18 # ratio of mass of Betelgeuse to mass of the sun (dimensionless)\nA2=A3*A4 # mass of Betelgeuse (pounds)\nA0=A2/A1 # number of queen size mattresses it would take to fill Betelgeuse (mattresses)```
"""

COMBINED_PROMPT = PromptTemplate(
    template=f"{SYSTEM_PROMPT_STRING}\n\n{FEW_SHOT_PROMPT_STRING}\n\n{INPUT_PROMPT.template}",
    input_variables=INPUT_PROMPT.input_variables,
)

def gen_prompt(example_dataset, k):
    examples = [
        {
            "input": INPUT_PROMPT.format(context=ex['context'], question=ex['question']),
            "output": OUTPUT_PROMPT.format(answer=ex['answer'], program=ex['program']),
        }
        for ex in example_dataset[:k]
    ]
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        suffix=INPUT_PROMPT.template,
        input_variables=INPUT_PROMPT.input_variables,
    )

    return prompt


class SamplePredictor:
    def __init__(self, temperature=0, prompt=COMBINED_PROMPT):

        self.prompt = prompt
        llm = OpenAI(
                temperature=temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

        self.llm_chain = LLMChain(prompt=self.prompt, llm=llm)

    @staticmethod
    def split_context_program(split):

        program = []
        context = []
        for segment in split[1:]:
            context_track = segment[0] == 'F'
            if context_track:
                context.append(segment)
            else:
                program.append(segment)
        program = '='.join(program[1:])
        context = 'CONTEXT:='+'='.join(context)
        answer = split[0]
        return answer, program, context

    def predict(self, dataset, N, include_context=True, verbose=False):
        direct_acc = 0
        compiled_acc = 0
        parsable = 0
        shuffled_idxs = np.arange(len(dataset))
        np.random.shuffle(shuffled_idxs)
        print("Shuffled Indices w/ fixed seed: ", shuffled_idxs[:10])
        predictions = []
        for i in tqdm(range(N)):
            entry = dataset[shuffled_idxs[i]]
            preds = self.llm_chain.run(context=entry["context"] if include_context else "CONTEXT:=", question=entry['question'])
            # preds = self.llm_chain.run(INPUT_PROMPT.format(context=entry['context'], question=entry['question']))

            try:
                answer = preds.split('=')[0]
                program = preds[preds.find("```python")+len("```python"):preds.rfind("```")]
                context = entry['context']

                loc = {}
                exec(program, globals(), loc)
                compiled_out = loc["A0"]
                compiled_units = program.split()[-1].replace('(', '').replace(')', '')

                prediction = {
                        "question": entry['question'],
                        "correct_answer": entry['answer'],
                        "direct_answer": answer,
                        "context": '; '.join(context.split('=')[1:]),
                        "program": program,
                        "raw_outputs": preds,
                        "compiled_answer": compiled_out,
                        "compiled_units": compiled_units
                    }

                # answer, program, context = self.split_context_program(preds.split("="))
                # compiled_answer = compile_fp(entry['context'], program)
                # compiled_out, compiled_units = convert_units(compiled_answer['P'])

                # prediction = {
                #         "question": entry['question'],
                #         "correct_answer": entry['answer'],
                #         "direct_answer": answer,
                #         "context": '; '.join(context.split('=')[1:]),
                #         "program": '; '.join(program.split('=')),
                #         "raw_outputs": preds,
                #         "compiled_answer": compiled_out,
                #         "compiled_units": compiled_units
                #     }

                direct_acc += accuracy_metric(entry['answer'], answer)
                compiled_acc += accuracy_metric(entry['answer'], compiled_out)
                parsable += 1
            except:
                prediction = {"question": entry['question'], "correct_answer": entry['answer'], "raw_outputs": preds}

            predictions.append(prediction)

            if verbose:
                print("Question: {}; \nCorrect Answer: {}".format(prediction["question"], prediction["correct_answer"]))

                if len(prediction) > 3:
                    print("Direct Answer is: {}\nCompiled Answer is: {} ({})\nSupporting Facts are: {}\nProgram:\n{}".format(prediction['direct_answer'], prediction['compiled_answer'], prediction['compiled_units'], prediction['context'], prediction['program']))
                else:
                    print("Unable to parse output; Outputting Raw Output:\n{}".format(prediction["raw_outputs"]))
        return direct_acc/N, compiled_acc/N, parsable/N, predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--no_context', action='store_true', help='Include context in the prompt')
    parser.add_argument('-D', action='store_true', help='Whether to use the distractor setting')
    parser.add_argument('--example_dataset', type=str, default="")
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('-k', type=int, default=5, help="Number of examples in the few-shot prompt")
    parser.add_argument('-N', type=int, default=-1, help="Number of test points")
    parser.add_argument('--temp', type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument('--verbose', action='store_true', help='Whether to print out the questions and answers')
    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == "":
        if (args.D):
            example_dataset = "./data/realFP/distractor_setting/train_distractor_realfp.json"
            dataset = "./data/realFP/distractor_setting/test_distractor_realfp.json"
        else:
            example_dataset = "./data/realFP/train_realfp.json"
            dataset = "./data/realFP/test_realfp.json"
    else:
        example_dataset = args.example_dataset
        dataset = args.dataset

    if (example_dataset[-5:] == '.json'):
        with open(example_dataset, 'rb') as f:
            example_dataset = json.load(f)

        prompt = gen_prompt(example_dataset, args.k)

    if (dataset[-5:] == '.json'):
        with open(dataset, 'rb') as f:
            dataset = json.load(f)


    predictor = SamplePredictor(temperature=args.temp)



    # print("Answering: {}".format(question))
    print("Evaluating on: {}".format(args.dataset))

    dir_acc, comp_acc, pars, _ = predictor.predict(dataset, args.N if args.N > 0 else len(dataset), include_context=(not args.no_context), verbose=args.verbose)

    print("Average Direct Accuracy: {}, Average Compiled Accuracy: {}, Parsable Percentage: {}".format(dir_acc, comp_acc, pars))

    # if len(prediction) > 1:
    #     print("Direct Answer is: {}\nCompiled Answer is: {} ({})\nSupporting Facts are: {}\nProgram: {}".format(prediction['direct_answer'], prediction['compiled_answer'], prediction['compiled_units'], prediction['context'], prediction['program']))
    # else:
    #     print("Unable to parse output; Outputting Raw Output:\n{}".format(prediction))
