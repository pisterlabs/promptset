import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets
from langchain import PromptTemplate


np.random.seed(42)
MODEL = "llama"


def prompt_formatter(template, context, user_instruct, query, answer_instruct,
                     answer="", model="vicuna", prompt_type="eval"
                     # system_msg="default"
                     ):

    # system_msg = "A chat between a curious user and an artificial intelligence assistant. The assistant follows user's instructions and gives helpful and direct answers to the user's questions." if system_msg == "default" else system_msg
    vicuna_kwargs = {
        "start": "",
        "instruction_begin": "",
        # "system_begin": "",
        # "system_end": "\n",
        "user": "USER:\n",
        "instruction_end": "\n",
        "assistant": "\nASSISTANT:\n",
        }

    llama_kwargs = {
        "start": "<s>",
        "instruction_begin": "[INST] ",
        # "system_begin": "<<SYS>>\n",
        # "system_end": "\n<</SYS>>\n",
        "user": "",
        "instruction_end": " [/INST] ",
        "assistant": "",
        }

    prompt_template = PromptTemplate(
        input_variables=[
            "start", "end", "instruction_begin", "instruction_end",
            "user", "user_instruct", "context", "query",
            "answer_instruct", "assistant", "answer", "end"
            # "system_begin", "system_end", "system_msg"
        ],
        template=template
    )

    current_kwargs = vicuna_kwargs if model == "vicuna" else llama_kwargs
    prompt = prompt_template.format(
        context=context,
        query=query,
        # system_msg=system_msg,
        user_instruct=user_instruct,
        answer_instruct=answer_instruct,
        answer="" if prompt_type == "eval" else answer,
        end="" if prompt_type == "eval" else " </s>",
        **current_kwargs
    )

    return prompt


def transform_pqa_train(sample):

    qa_template = "{start}{instruction_begin}{user}{user_instruct}{context}{query}{answer_instruct}{instruction_end}{assistant}{answer}{end}"
    task_instructions = {
        "prompt_1": "# Carefully read and consider the following <Context>:\n",
    }

    # task_probabilities = {
    #     "simple_answer": 0.3,
    #     "complex_answer": 0.5,
    #     "keyword_generation": 0.2
    # }

    answer_instruct = ("# Start your response with either 'yes' or 'no', "
                       "followed by a short, concise explanation of your "
                       "answer based on the <Context>. \nYour answer is: ")
    query = (f"# Based on all the provided <Context>, answer the following "
             f"<Question>: \n## Question: \n{sample['question']}\n\n")
    full_answer = sample['final_decision'].capitalize() + '. ' + sample['long_answer']  # "Based on the context provided, the answer is " + ...
    context_list = sample['context']['contexts']
    formatted_context = '\n'.join([f"### {context}" for context in context_list])
    contexts = f"## Context\n{formatted_context}\n\n"

    # tasks = list(task_instructions.keys())
    # probabilities = list(task_probabilities.values())
    # selected_task = np.random.choice(tasks, p=probabilities)

    # if selected_task == "simple_answer":
    #     query = "\nQuestion: " + question + "\n"
    #     answer = full_answer
    # elif selected_task == "complex_answer":
    #     query = "\nQuestion: " + question + "\n"
    #     answer = "Answer: " + decision + ". " + explaination
    # else:
    #     query = ""
    #     answer = "Keywords: " + keywords

    conversation = prompt_formatter(
        template=qa_template,
        context=contexts,
        user_instruct=task_instructions["prompt_1"],
        query=query,
        answer=full_answer,
        answer_instruct=answer_instruct,
        model=MODEL,
        prompt_type="train"
    )

    return {'conversation': conversation}


def transform_pqa_val(sample):
    qa_template = "{start}{instruction_begin}{user}{user_instruct}{context}{query}{answer_instruct}{instruction_end}{assistant}{answer}{end}"
    instruction = "# Carefully read and consider the following <Context>:\n"
    answer_instruct = ("# Start your response with either 'yes' or 'no', "
                       "followed by a short, concise explanation of your "
                       "answer based on the <Context>. \nYour answer is: ")
    query = (f"# Based on all the provided <Context>, answer the following "
             f"<Question>: \n## Question: \n{sample['question']}\n\n")
    context_list = sample['context']['contexts']
    formatted_context = '\n'.join([f"### {context}" for context in context_list])
    contexts = f"## Context\n{formatted_context}\n\n"
    decision = sample['final_decision']
    full_answer = sample['final_decision'].capitalize() + '. ' + sample['long_answer']

    conversation = prompt_formatter(
        template=qa_template,
        context=contexts,
        user_instruct=instruction,
        query=query,
        answer=full_answer,
        answer_instruct=answer_instruct,
        model=MODEL,
        prompt_type="eval"
    )
    return {'conversation': conversation,  # + "Based on the context provided, the answer is "
            'answer': decision,
            'full_answer': full_answer}


# Load the datasets.
pqa_art = load_dataset('pubmed_qa', 'pqa_artificial')
pqa_l = load_dataset('pubmed_qa', 'pqa_labeled')

# Filter and shuffle the datasets.
# Train.
pqa_l = pqa_l['train'].shuffle(seed=42).filter(
    lambda sample: sample['final_decision'] != "maybe"
    )

# Validation.
pqa_art_yes = pqa_art['train'].shuffle(seed=42).filter(
    lambda sample: sample['final_decision'] == "yes"
    ).select(range(50))
pqa_art_no = pqa_art['train'].shuffle(seed=42).filter(
    lambda sample: sample['final_decision'] == "no"
    ).select(range(50))
pqa_art = concatenate_datasets([pqa_art_yes, pqa_art_no])

# Transform the datasets.
# Train.
pqa_l = pqa_l.map(transform_pqa_train).select_columns(['conversation'])
# Validation.
pqa_art = pqa_art.map(transform_pqa_val).select_columns(
    ['conversation', 'answer', 'full_answer']
    ).shuffle(seed=42)

# Concatenate and save the datasets.
merged_dataset = DatasetDict({
    "train": pqa_l,
    "test": pqa_art,
})

merged_dataset.save_to_disk(f"data/finetuning_{MODEL}")
