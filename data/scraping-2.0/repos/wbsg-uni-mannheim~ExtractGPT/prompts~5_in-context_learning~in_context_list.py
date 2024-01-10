import json
import random
from datetime import datetime
from json import JSONDecodeError

import click
import torch
from dotenv import load_dotenv
from langchain import LLMChain, HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import SystemMessage
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from pieutils import create_pydanctic_models_from_known_attributes, save_populated_task, parse_llm_response_to_json
from pieutils.evaluation import evaluate_predictions
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_test_set
from pieutils.search import CategoryAwareMaxMarginalRelevanceExampleSelector, CategoryAwareRandomExampleSelector, \
    CategoryAwareFixedExampleSelector, CategoryAwareSemanticSimilarityExampleSelector, \
    CategoryAwareSemanticSimilarityDifferentAttributeValuesExampleSelector


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=False, help='Verbose mode')
@click.option('--shots', default=3, help='Number of shots for few-shot learning')
@click.option('--example_selector', default='MaxMarginalRelevance', help='Example selector for few-shot learning')
@click.option('--train_percentage', default=1.0, help='Percentage of training data used for in-context learning')
def main(dataset, model, verbose, shots, example_selector, train_percentage):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    task_dict['task_name'] = "multiple_attribute_values-great-incontext-{}-{}-shots".format(dataset, shots)
    task_dict['task_prefix'] = "Extract the attribute values from the product {part} below in a JSON format. Valid " \
                                "attributes are {attributes}. If an attribute is not present in the product title, " \
                                "the attribute value is supposed to be 'n/a'."

    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_dict['train_percentage'] = train_percentage

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict)

    # Initialize model
    task_dict['model'] = model
    if 'gpt-3' in task_dict['model'] or 'gpt-4' in task_dict['model']:
        llm = ChatOpenAI(model_name=task_dict['model'], temperature=0)
    else:
        if 'StableBeluga' in task_dict['model']:
            tokenizer = AutoTokenizer.from_pretrained(task_dict['model'], use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(task_dict['model'], torch_dtype=torch.float16,
                                                         low_cpu_mem_usage=True, device_map="auto")
            model.tie_weights()

            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0, top_p=0.95)
            llm = HuggingFacePipeline(pipeline=pipe)
        elif 'SOLAR' in task_dict['model']:
            tokenizer = AutoTokenizer.from_pretrained(task_dict['model'])
            model = AutoModelForCausalLM.from_pretrained(
                task_dict['model'],
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True
                #rope_scaling={"type": "dynamic", "factor": 2}  # allows handling of longer inputs
            )

            model.tie_weights()

            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0, use_cache=True)
            llm = HuggingFacePipeline(pipeline=pipe)
        else:
            print('Unknown model!')

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])
    # Create chains to extract attribute values from product titles
    system_setting = SystemMessage(
        content='You are a world-class algorithm for extracting information in structured formats. ')
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")

    # Add few shot sampling
    if example_selector == 'MaxMarginalRelevance':
        category_example_selector = CategoryAwareMaxMarginalRelevanceExampleSelector(task_dict['dataset_name'],
                                                                                     list(task_dict['known_attributes'].keys()),
                                                                                     load_from_local=True, k=shots)
    elif example_selector == 'SemanticSimilarity':
        category_example_selector = CategoryAwareSemanticSimilarityExampleSelector(task_dict['dataset_name'],
                                                                                     list(task_dict['known_attributes'].keys()),
                                                                                     load_from_local=True, k=shots)
    elif example_selector == 'SemanticSimilarityDifferentAttributeValues':
        category_example_selector = CategoryAwareSemanticSimilarityDifferentAttributeValuesExampleSelector(task_dict['dataset_name'],
                                                                                 list(task_dict['known_attributes'].keys()),
                                                                                 category_2_pydantic_models=pydantic_models,
                                                                                 load_from_local=True, k=shots, train_percentage=train_percentage)
    elif example_selector == 'Random':
        category_example_selector = CategoryAwareRandomExampleSelector(task_dict['dataset_name'],
                                                                       list(task_dict['known_attributes'].keys()),
                                                                       k=shots)
    elif example_selector == 'Fixed':
        category_example_selector = CategoryAwareFixedExampleSelector(task_dict['dataset_name'],
                                                                                 list(task_dict['known_attributes'].keys()),
                                                                                 category_2_pydantic_models=pydantic_models,
                                                                                 load_from_local=True, k=shots)
    else:
        raise NotImplementedError("Example selector {} not implemented".format(example_selector))

    # Define the few-shot prompt.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input", "category"],
        example_selector=category_example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )

    prompt = ChatPromptTemplate(messages=[system_setting, human_task_prompt, few_shot_prompt,
                                          human_task_prompt, human_message_prompt])

    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )

    def select_and_run_llm(category, input_text, part='title'):
        pred = None
        try:
            response = chain.run(input=input_text, attributes=', '.join(task_dict['known_attributes'][category]),
                             category=category, part=part)
            #if verbose:
            #    print(response)
            try:
                # Convert json string into pydantic model
                json_response = json.loads(response)
                if verbose:
                    print(json_response)
                pred = pydantic_models[category](**json.loads(response))
            except ValidationError as valError:
                print(valError)
            except JSONDecodeError as e:
                converted_response = parse_llm_response_to_json(response)
                if verbose:
                    print('Converted response: ')
                    print(converted_response)
                try:
                    pred = pydantic_models[category](**converted_response)
                except ValidationError as valError:
                    print(valError)
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)
                print('Response: ')
                print(response)
        except Exception as e:
            print(e)


        return pred

    with get_openai_callback() as cb:
        # Mave dataset has multiple sources. We need to iterate over all of them.
        if dataset == 'mave':
            part_predictions = {}
            parts = ['title', 'description', 'features', 'attributes']
            # Run LLM for each part of the product
            for part in parts:
                example_part = part if part != 'title' else 'input'
                example_part = example_part if example_part != 'attributes' else 'rest'
                preds = [select_and_run_llm(example['category'], example[example_part], part=part) for example in
                         tqdm(task_dict['examples'], disable=not verbose)]
                part_predictions[example_part] = preds

            # Combine all predictions using a majority vote
            preds = [fuse_models(pydantic_models[example['category']], pred_title, pred_description, pred_features,
                                 pred_rest)
                     for example, pred_title, pred_description, pred_features, pred_rest
                     in zip(task_dict['examples'], part_predictions['input'], part_predictions['description'],
                            part_predictions['features'], part_predictions['rest'])]

        # Mave dataset has multiple sources. We need to iterate over all of them.
        elif dataset == 'mave_v2':
            hashed_predictions = {category: {} for category in task_dict['known_attributes']} # Cache predictions per category
            preds = []
            # Run LLM for each part of the product
            for example in tqdm(task_dict['examples'], disable=not verbose):
                example_preds = []
                hashed_input = hash(example['input'])
                if hashed_input in hashed_predictions[example['category']]:
                    example_preds.append(hashed_predictions[example['category']][hashed_input])
                    print("Cached")
                    continue
                title_prediction = select_and_run_llm(example['category'], example['input'], part='title')
                hashed_predictions[example['category']][hashed_input] = title_prediction
                example_preds.append(title_prediction)

                for description in example['paragraphs']:
                    hashed_description = hash(description)
                    if hashed_description in hashed_predictions[example['category']]:
                        example_preds.append(hashed_predictions[example['category']][hashed_description])
                        print("Cached")
                        continue

                    description_prediction = select_and_run_llm(example['category'], description, part='description')
                    hashed_predictions[example['category']][hashed_description] = description_prediction
                    example_preds.append(description_prediction)

                pred = fuse_models(pydantic_models[example['category']], *example_preds)
                if verbose:
                    print(pred)
                preds.append(pred)

        else:
            preds = [select_and_run_llm(example['category'], example['input']) for example in
                     tqdm(task_dict['examples'], disable=not verbose)]

        task_dict['total_tokens'] = cb.total_tokens
        print(f"Total Tokens: {cb.total_tokens}")


    # Calculate recall, precision and f1
    task_dict['results'] = evaluate_predictions(preds, task_dict)
    # Save populated task and make sure that it is saved in the correct folder!
    save_populated_task(task_dict['task_name'], task_dict)

if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
