import json
# Update Task Dict
import random
from datetime import datetime
from json import JSONDecodeError

import click
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    AIMessagePromptTemplate
from pydantic import ValidationError
from tqdm import tqdm

from pieutils import save_populated_task, create_pydanctic_model_from_pydantic_meta_model, \
    create_dict_of_pydanctic_product, convert_to_json_schema, create_pydanctic_models_from_known_attributes
from pieutils.evaluation import evaluate_predictions
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_test_set, load_known_attribute_values
from pieutils.pydantic_models import ProductCategory


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--verbose', default=False, help='Verbose mode')
@click.option('--train_percentage', default=1.0, help='Percentage of training data to use')
def main(dataset, verbose, train_percentage):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    task_dict['task_name'] = "attribute_value_lookup"
    task_dict['model'] = 'attribute_value_lookup'
    task_dict['task_prefix'] = ""
    task_dict['data_source'] = dataset
    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict)
    known_attribute_values = load_known_attribute_values(task_dict['dataset_name'], n_examples=9999, train_percentage=train_percentage)

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])

    def apply_attribute_value_lookup(category, example, attribute_values):
        pred_dict = {}
        for attribute in attribute_values[category]:
            pred_dict[attribute] = 'n/a'
            for attribute_value in attribute_values[category][attribute]:
                if attribute_value in example:
                    pred_dict[attribute] = attribute_value
                    break
        pred = pydantic_models[category](**pred_dict)
        return pred



    # Mave dataset has multiple sources. We need to iterate over all of them.
    if dataset == 'mave':
        part_predictions = {}
        parts = ['title', 'description', 'features', 'attributes']
        # Run LLM for each part of the product
        for part in parts:
            example_part = part if part != 'title' else 'input'
            example_part = example_part if example_part != 'attributes' else 'rest'
            preds = [apply_attribute_value_lookup(example['category'], example[example_part], known_attribute_values) for example in
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
            title_prediction = apply_attribute_value_lookup(example['category'], example['input'], known_attribute_values)
            hashed_predictions[example['category']][hashed_input] = title_prediction
            example_preds.append(title_prediction)

            for description in example['paragraphs']:
                hashed_description = hash(description)
                if hashed_description in hashed_predictions[example['category']]:
                    example_preds.append(hashed_predictions[example['category']][hashed_description])
                    print("Cached")
                    continue

                description_prediction = apply_attribute_value_lookup(example['category'], description, known_attribute_values)
                hashed_predictions[example['category']][hashed_description] = description_prediction
                example_preds.append(description_prediction)

            pred = fuse_models(pydantic_models[example['category']], *example_preds)
            if verbose:
                print(pred)
            preds.append(pred)

    else:
        preds = [apply_attribute_value_lookup(example['category'], example['input'], known_attribute_values) for example in
                 tqdm(task_dict['examples'], disable=not verbose)]

    # Calculate recall, precision and f1
    task_dict['results'] = evaluate_predictions(preds, task_dict)
    # Save populated task and make sure that it is saved in the correct folder!
    save_populated_task(task_dict['task_name'], task_dict)


if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
