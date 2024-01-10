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
    FewShotChatMessagePromptTemplate
from pydantic import ValidationError
from tqdm import tqdm

from pieutils import save_populated_task, create_pydanctic_model_from_pydantic_meta_model, \
    create_dict_of_pydanctic_product, convert_to_json_schema, create_tabular_pydanctic_model_from_pydantic_meta_model
from pieutils.evaluation import evaluate_predictions
from pieutils.preprocessing import update_task_dict_from_test_set, load_known_attribute_values
from pieutils.pydantic_models import ProductCategory
from pieutils.search import cluster_examples_using_kmeans, CategoryAwareMaxMarginalRelevanceExampleSelector


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=False, help='Verbose mode')
@click.option('--shots', default=3, help='Number of shots for few-shot learning')
def main(dataset, model, verbose, shots):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    task_dict['task_name'] = "chatgpt_tabluar_extraction_sorted_attributes"
    task_dict['task_prefix'] = "Split the product title by whitespace. " \
                               "Extract the valid attribute values from the product titles in JSON format. All " \
                               "valid attributes are provided in the JSON schema. Unknown attribute values should be" \
                               " marked as n/a. "

    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict)
    known_attribute_values = load_known_attribute_values(task_dict['dataset_name'])

    # Initialize model
    task_dict['model'] = model
    llm = ChatOpenAI(model_name=model, temperature=0)

    # Ask LLM to generate meta models for each product category and attribute

    # Create Chains
    system_message_prompt = SystemMessagePromptTemplate.from_template("You are a world class algorithm for creating "
                                                                      "descriptions of product categories and their "
                                                                      "attributes following this JSON schema: \n {"
                                                                      "schema}.")
    human_task_meta_model = HumanMessagePromptTemplate.from_template("Write short descriptions for the product category"
                                                                     "{category} and the attributes {attributes} that "
                                                                     "are helpful to identify relevant attribute "
                                                                     "values in product titles. "
                                                                     "The descriptions should not be longer than one "
                                                                     "sentence. "
                                                                     "The following attribute values are known for "
                                                                     "each attribute: \n "
                                                                     "{known_attribute_values}. \n"
                                                                     "Respond with a JSON object following the "
                                                                     "provided schema.")
    prompt_meta_model = ChatPromptTemplate(messages=[system_message_prompt, human_task_meta_model])

    pydantic_models = {}
    pydantic_single_models = {}
    models_json = {}
    for category in task_dict['known_attributes']:
        print('Create model for category: {}'.format(category))

        chain = LLMChain(
            prompt=prompt_meta_model,
            llm=llm,
            verbose=verbose
        )
        # chain_meta_model = create_structured_output_chain(ProductCategory, llm, prompt_meta_model, verbose=True)
        known_attribute_values_per_category = json.dumps(known_attribute_values[category])
        response = chain.run({'schema': convert_to_json_schema(ProductCategory, False), 'category': category,
                              'attributes': ', '.join(task_dict['known_attributes'][category]),
                              'known_attribute_values': known_attribute_values_per_category})
        pred = ProductCategory(**json.loads(response))
        print(pred)
        pydantic_models[category] = create_tabular_pydanctic_model_from_pydantic_meta_model(pred)
        pydantic_single_models[category] = create_pydanctic_model_from_pydantic_meta_model(pred)
        models_json[category] = create_dict_of_pydanctic_product(pydantic_models[category])

    llm = ChatOpenAI(model_name=task_dict['model'], temperature=0)

    # Persist models
    with open('prompts/meta_models/models_by_{}_{}.json'.format(task_dict['task_name'], task_dict['model'],
                                                              task_dict['dataset_name']), 'w', encoding='utf-8') as f:
        json.dump(models_json, f, indent=4)

        # Create Chains
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are a world class algorithm for extracting information in structured formats. \n {schema} ")
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])

    # Add few shot sampling
    category_example_selector = CategoryAwareMaxMarginalRelevanceExampleSelector(task_dict['dataset_name'],
                                                                                 list(task_dict[
                                                                                          'known_attributes'].keys()),
                                                                                 category_2_pydantic_models=pydantic_models,
                                                                                 load_from_local=False, k=shots,
                                                                                 tabular=True)

    # Define the few-shot prompt.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input", "category"],
        example_selector=category_example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{inputs}"), ("ai", "{outputs}")]
        ),
    )

    human_message_prompt = HumanMessagePromptTemplate.from_template("{input_records}")
    prompt_list = [system_message_prompt, human_task_prompt, few_shot_prompt, human_task_prompt, human_message_prompt]
    prompt = ChatPromptTemplate(messages=prompt_list)

    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )

    # Cluster examples by category and groups of up to 5 examples
    for category in task_dict['known_attributes']:
        print('Cluster examples for category: {}'.format(category))
        examples_per_category = [example for example in task_dict['examples'] if example['category'] == category]
        clustered_examples = cluster_examples_using_kmeans(examples_per_category, 5)
        clustered_examples_dict = {clustered_example['input']: clustered_example['cluster_id'] for clustered_example in
                                   clustered_examples}
        for example in task_dict['examples']:
            if example['category'] != category:
                continue
            example['cluster_id'] = clustered_examples_dict[example['input']]

    def select_and_run_llm(category, input_texts):
        """Select the correct LLM for the category and run it on the input texts"""
        pred = [None for i in range(len(input_texts))]
        #if len(input_texts) > 5:
        #    print('Too many input texts for LLM. Only the first 5 will be used.')
        try:
            response_llm = chain.run(input_records='\n'.join(input_texts), input=input_texts[0],
                                 schema=convert_to_json_schema(pydantic_models[category]), category=category)

            try:
                # Convert json string into pydantic model
                json_response = json.loads(response_llm)
                print(json_response)
                preds = pydantic_models[category](**json.loads(response_llm))
                if len(input_texts) != len(preds.records):

                    while len(input_texts) > len(preds.records):
                        print('Missing predictions for some input texts.')
                        preds.records.append(None)
                    if len(input_texts) < len(preds.records):
                        print('Too many predictions.')
                        preds.records = preds.records[:len(input_texts)]
                pred = preds.records
            except ValidationError as valError:
                if len(input_texts) == 1:
                    try:
                        pred = [pydantic_single_models[category](**json.loads(response_llm))]
                    except ValidationError as valError:
                        print(valError)
            except JSONDecodeError as jsonError:
                print(jsonError)
            except Exception as e:
                # Happens if a list instead of the expected object is returned by the LLM
                print(e)
        except Exception as e:
            print(e)

        return pred

    # Sort examples by category and cluster id
    task_dict['examples'] = sorted(task_dict['examples'], key=lambda x: (x['category'], x['cluster_id']))
    # Run LLM on examples
    with get_openai_callback() as cb:
        # Chunk examples by category and cluster id
        current_category = task_dict['examples'][0]['category']
        current_cluster_id = task_dict['examples'][0]['cluster_id']
        current_examples = []
        preds = []
        for example in tqdm(task_dict['examples'],disable=not verbose):
            if example['category'] != current_category or example['cluster_id'] != current_cluster_id:
                # Run LLM on current examples
                new_preds = select_and_run_llm(current_category, current_examples)
                preds.extend(new_preds)
                # Reset current examples
                current_examples = []
                current_category = example['category']
                current_cluster_id = example['cluster_id']
            current_examples.append(example['input'])
        # Run LLM on last examples
        new_preds = select_and_run_llm(current_category, current_examples)
        preds.extend(new_preds)

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
