import tqdm
import random
from dotenv import dotenv_values
from langchain import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import load_cpa_dataset, load_cta_dataset, save_pickle_file, textada_embeddings
import os

if __name__ == "__main__":

    # Load env file with API KEY using full path
    config = dotenv_values("/full/path/to/file/key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

    # StableBeluga7B
    model_name = "stabilityai/StableBeluga-7B"
    mod = "stablebeluga7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="hf_cache/")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir="hf_cache/")

    # SOLAR
    # model_name = "upstage/SOLAR-0-70b-16bit"
    # mod = "solar"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="hf_cache/")
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, load_in_8bit=True, device_map="auto", cache_dir="/ceph/kkorini/hf_cache/", temperature=0, do_sample=True)

    system_messages_tasks = ["S1", "S2", "S3", "S4", "S5"]
    system_messages_content = {
        "S1": "Generate knowledge facts about some terms that can help in the task of column type annotation. Reply with only one sentence! ",
        "S5": "Generate knowledge facts about some terms that can help in the task of column type annotation and one example value for the term. Reply with only one sentence!",
        "S2": "Generate definitions about some terms that can help in the task of column type annotation. Reply with only one sentence!",
        "S3": "Generate descriptions about some terms that can help in the task of column type annotation. Reply with only one sentence!",
        "S4": "Perform column type annotation, e.g. annotate each column of a table with a label that captures the meaning of the values of the column. As preparation for this task, you are asked to generate definitions of the labels. The definitions should be helpful for distinguishing different labels. Reply with only one sentence!",
    }

    instruction_messages = ["I1", "I2"]
    instruction_messages_content = {
        "I1": "1. Look at the input given to you and make tables out of them. 2. The first row of each table are the column types of each column. 3. Look at their statistical and semantic characteristics of the columns. 4.Generate knowledge for the required term by looking at the whole table. 5. Do not generate specific knowledge for each of the columns. 6.Reply only with knowledge facts not examples.",
        "I2": "1. Look at the input given to you and make tables out of them. 2. The first row of each table are the column types of each column. 3. Look at their statistical and semantic characteristics of the columns. 4.Generate a definition of the term by looking at the whole table. 5. Do not generate specific knowledge for each of the columns. 6.Reply only with knowledge facts not examples.",
    }

    # A-prompts
    general_prompts = ["A1", "A2", "A3", "A4", "A5", "A6"]
    general_prompts_content = {
        "A1": "Generate some knowledge facts about the term ",
        "A2": "How can you distinguish if some values are about the term ",
        "A3": "Generate some rules to follow if you need to decide if some values are about the term ",
        "A4": "Generate some instructions to follow if you need to decide if some values are about the term ",
        "A5": "What do you know about the term ",
        "A6": "Generate some instructions to follow if you need to decide if some values may belong to the term ",
    }
    
    # B-prompts
    table_prompts = ["TB1", "TB2", "TB3", "TB4", "TB5", "TB6", "TB7"]
    table_prompts_content = {
        "TB1": "Generate some knowledge facts about the term ",
        "TB2": "What characteristics can you learn about the term ",
        "TB3": "Learn about the term ",
        "TB4": "Generate some rules to follow if you need to decide if some values are about the term ",
        "TB5": "Generate some instructions to follow if you need to decide if some values are about the term ",
        "TB6": "What semantic characteristics and statistics can you learn about the term ",
        "TB7": "What value patterns can you learn about the term ",
    }

    a_template = """Task: {task}
    {gen}{label}."""

    tb_template = """Task: {task}
    Instructions: {instructions}
    {gen}{label} using the following examples:

    {examples}
    What did you learn about {label}?"""

    # CTA generation
    for dataset in ["sotabv2", "t2dv2-webtables","sportstables"]:
        examples, labels, test_table_type_labels, train_examples, train_example_labels, train_table_type_labels, labels_to_text = load_cta_dataset(dataset,"-for-kg")
        all_labels = [labels_to_text[l] for l in labels_to_text]

        # Run A prompts with combination of S messages:
        for system_mess in system_messages_tasks:
            for g in general_prompts:
                print(f"{system_mess}_{g}_prompt_knowledge")
                if f"{system_mess}_{g}_prompt_knowledge.pkl" not in os.listdir(f"knowledge/{mod}/{dataset}/"):
                    prompts = []
                    prompt = PromptTemplate(template=a_template, input_variables=['task', 'label', 'gen'])
                    
                    for label in all_labels:
                        text_prompt = prompt.format(task=system_messages_content[system_mess], gen=general_prompts_content[g], label=label)
                        prompts.append(text_prompt)
                    
                    model_answers = []
                    for prompt in tqdm.tqdm(prompts, total=len(prompts)):
                        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                        output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)
                        model_answers.append(tokenizer.decode(output[0], skip_special_tokens=True))

                    save_pickle_file(f"knowledge/{mod}/{dataset}/{system_mess}_{g}_prompt_knowledge.pkl", model_answers)
                    save_pickle_file(f"knowledge/{mod}/{dataset}/{system_mess}_{g}_prompt_knowledge-prompts.pkl", prompts)
                    definitions = [answer.replace(prompts[i],"") for i, answer in enumerate(model_answers)]
                    save_pickle_file(f"embeddings/{mod}/{system_mess}_{g}_knowledge_embeddings_{dataset}.pkl", textada_embeddings(definitions, OPENAI_API_KEY))
                else:
                    print(f"knowledge/{mod}/{dataset}/{system_mess}_{g}_prompt_knowledge.pkl")
        
        # Run B prompts with combination of S and I messages:
        for system_mess in system_messages_tasks:
            for instructions in instruction_messages:
                for tab in table_prompts:
                    print(f"{system_mess}_{instructions}_{tab}_prompt_knowledge")
                    if f"{system_mess}_{instructions}_{tab}_prompt_knowledge.pkl" not in os.listdir(f"knowledge/{mod}/{dataset}/"):
                        prompts = []
                        prompt = PromptTemplate(template=tb_template, input_variables=['task', 'instructions', 'label', 'gen', 'examples'])

                        for label in all_labels:
                            random_examples = """"""

                            for i in range(0,3):
                                index = random.choice([j for j, e in enumerate(train_example_labels) if label in e])
                                random_examples += f"""{train_examples[index]}\n"""

                            random_examples = random_examples.strip()

                            text_prompt = prompt.format(task=system_messages_content[system_mess], gen=table_prompts_content[tab], instructions=instruction_messages_content[instructions], label=label, examples=random_examples)
                            prompts.append(text_prompt)

                        model_answers = []
                        for prompt in tqdm.tqdm(prompts, total=len(prompts)):
                            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                            output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)
                            model_answers.append(tokenizer.decode(output[0], skip_special_tokens=True))

                        save_pickle_file(f"knowledge/{mod}/{dataset}/{system_mess}_{instructions}_{tab}_prompt_knowledge.pkl", model_answers)
                        save_pickle_file(f"knowledge/{mod}/{dataset}/{system_mess}_{instructions}_{tab}_prompt_knowledge-prompts.pkl", prompts)
                        definitions = [answer.replace(prompts[i],"") for i, answer in enumerate(model_answers)]
                        save_pickle_file(f"embeddings/{mod}/{system_mess}_{instructions}_{tab}_knowledge_embeddings_{dataset}.pkl", textada_embeddings(definitions, OPENAI_API_KEY))
                    else:
                        print(f"knowledge/{mod}/{dataset}/{system_mess}_{instructions}_{tab}_prompt_knowledge.pkl")


    # CPA generation
    system_messages_tasks = ["S1", "S2", "S3", "S4", "S5"]
    system_messages_content = {
        "S1": "Generate knowledge facts about some terms that can help in the task of column relationship prediction. Reply with only one sentence! ",
        "S5": "Generate knowledge facts about some terms that can help in the task of column relationship prediction and one example value for the term. Reply with only one sentence!",
        "S2": "Generate definitions about some terms that can help in the task of column relationship prediction. Reply with only one sentence!",
        "S3": "Generate descriptions about some terms that can help in the task of column relationship prediction. Reply with only one sentence!",
        "S4": "Perform column property annotation, e.g. annotate the relationships of columns with a label that captures the relationship. As preparation for this task, you are asked to generate definitions of the relationship labels. The definitions should be helpful for distinguishing different labels. Reply with only one sentence!",
    }

    instruction_messages = ["I1", "I2"]
    instruction_messages_content = {
        "I1": "1. Look at the input given to you and make tables out of them. 2. The first row of each table are the column relationships of each column with the first column of the table. 3. Look at their statistical and semantic characteristics of the columns. 4.Generate knowledge for the required term by looking at the whole table. 5. Do not generate specific knowledge for each of the columns. 6.Reply only with knowledge facts not examples.",
        "I2": "1. Look at the input given to you and make tables out of them. 2. The first row of each table are the column relationships of each column with the first column of the table. 3. Look at their statistical and semantic characteristics of the columns. 4.Generate a definition for the required term by looking at the whole table. 5. Do not generate specific knowledge for each of the columns. 6.Reply only with knowledge facts not examples.",
    }

    for dataset in ["sotabv2", "t2dv2-webtables"]:
        examples, labels, test_table_type_labels, train_examples, train_example_labels, train_table_type_labels, labels_to_text, text_to_label, labels_joined, train, test = load_cpa_dataset(dataset,"-for-kg",False)
        all_labels = [labels_to_text[l] for l in labels_to_text]

        # Run A prompts with combination of S messages:
        for system_mess in system_messages_tasks:
            for g in general_prompts:
                print(f"cpa-{system_mess}_{g}_prompt_knowledge")
                if f"cpa-{system_mess}_{g}_prompt_knowledge.pkl" not in os.listdir(f"knowledge/{mod}/{dataset}/"):
                    prompts = []
                    prompt = PromptTemplate(template=a_template, input_variables=['task', 'label', 'gen'])
                    
                    for label in all_labels:
                        text_prompt = prompt.format(task=system_messages_content[system_mess], gen=general_prompts_content[g], label=label)
                        prompts.append(text_prompt)
                    
                    model_answers = []
                    for prompt in tqdm.tqdm(prompts, total=len(prompts)):
                        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                        output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)
                        model_answers.append(tokenizer.decode(output[0], skip_special_tokens=True))

                    save_pickle_file(f"knowledge/{mod}/{dataset}/cpa-{system_mess}_{g}_prompt_knowledge.pkl", model_answers)
                    save_pickle_file(f"knowledge/{mod}/{dataset}/cpa-{system_mess}_{g}_prompt_knowledge-prompts.pkl", prompts)
                    definitions = [answer.replace(prompts[i],"") for i, answer in enumerate(model_answers)]
                    save_pickle_file(f"embeddings/{mod}/cpa-{system_mess}_{g}_knowledge_embeddings_{dataset}.pkl", textada_embeddings(definitions, OPENAI_API_KEY))
                else:
                    print(f"knowledge/{mod}/{dataset}/cpa-{system_mess}_{g}_prompt_knowledge.pkl")
        
        # Run B prompts with combination of S and I messages:
        for system_mess in system_messages_tasks:
            for instructions in instruction_messages:
                for tab in table_prompts:
                    print(f"cpa-{system_mess}_{instructions}_{tab}_prompt_knowledge")
                    if f"cpa-{system_mess}_{instructions}_{tab}_prompt_knowledge.pkl" not in os.listdir(f"knowledge/{mod}/{dataset}/"):
                        prompts = []
                        prompt = PromptTemplate(template=tb_template, input_variables=['task', 'instructions', 'label', 'gen', 'examples'])

                        for label in all_labels:
                            random_examples = """"""

                            for i in range(0,3):
                                index = random.choice([j for j, e in enumerate(train_example_labels) if label in e])
                                random_examples += f"""{train_examples[index]}\n"""

                            random_examples = random_examples.strip()

                            text_prompt = prompt.format(task=system_messages_content[system_mess], gen=table_prompts_content[tab], instructions=instruction_messages_content[instructions], label=label, examples=random_examples)
                            prompts.append(text_prompt)

                        model_answers = []
                        for prompt in tqdm.tqdm(prompts, total=len(prompts)):
                            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                            output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)
                            model_answers.append(tokenizer.decode(output[0], skip_special_tokens=True))

                        save_pickle_file(f"knowledge/{mod}/{dataset}/cpa-{system_mess}_{instructions}_{tab}_prompt_knowledge.pkl", model_answers)
                        save_pickle_file(f"knowledge/{mod}/{dataset}/cpa-{system_mess}_{instructions}_{tab}_prompt_knowledge-prompts.pkl", prompts)
                        definitions = [answer.replace(prompts[i],"") for i, answer in enumerate(model_answers)]
                        save_pickle_file(f"embeddings/{mod}/cpa-{system_mess}_{instructions}_{tab}_knowledge_embeddings_{dataset}.pkl", textada_embeddings(definitions, OPENAI_API_KEY))
                    else:
                        print(f"knowledge/{mod}/{dataset}/cpa-{system_mess}_{instructions}_{tab}_prompt_knowledge.pkl")