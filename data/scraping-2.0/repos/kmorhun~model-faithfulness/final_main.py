import os
import openai
from flan_call import get_hugg_completion
from gpt_api import get_gpt_completion, gpt_save_output_json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import google.generativeai as palm
from dotenv import load_dotenv
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from transformers import pipeline
from prompts import *
from preprocess import *
from datetime import datetime

prompt_1 = create_prompt("data/inputs/examples.json", PROMPT_TEMPLATE_FEW_SHOT_RESTAURANT, FORMATTED_EXAMPLE_TEMPLATE_1, NEW_EXAMPLE_JSON_TEMPLATE, ["example_1", "example_2", "example_3"], PANCAKES_EXAMPLE)
prompt_2 = create_prompt("data/inputs/examples.json", PROMPT_TEMPLATE_FEW_SHOT_RESTAURANT, FORMATTED_EXAMPLE_TEMPLATE_1, NEW_EXAMPLE_JSON_TEMPLATE, ["example_1", "example_2"], PANCAKES_EXAMPLE)
prompt_3 = create_prompt("data/inputs/examples.json", PROMPT_TEMPLATE_ZERO_SHOT_RESTAURANT, None, NEW_EXAMPLE_JSON_TEMPLATE, [], PANCAKES_EXAMPLE, zero_shot=True)

def extract_review(prompt):
    #this gets the part of the prompt after "Human: "
    review_assistant = prompt.split("Human: ")[-1]
    # print("review_assistant ", review_assistant, "\n")

    #this gets the actual review
    review = review_assistant.split("\n")[0]
    # print("review ", review, "\n")

    return review

def run_gpt(dataset, few_shot_example_names, category_name):
    """
    Prompts GPT-4, and reformats the model's response into a json file, which gets saved to data/outputs
    As a checkpoint, it also saves the intermediate step from model response to a file in data/outputs

    Takes about 3-4 seconds per review
    """
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key

    # Check if the "outputs" folder exists, and create it if it doesn't
    outputs_folder = "data/outputs"
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    # Check if the "logs" folder exists, and create it if it doesn't
    logs_folder = "logs"
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    responses_json = {"responses": []}
    new_reviews = []
    count_processed = 0

    for review_name, review in dataset.items():
        review_text = review["review"]

        # create a prompt based on this review
        if (category_name == "restaurant"):
            prompt = create_prompt("data/inputs/examples.json", PROMPT_TEMPLATE_FEW_SHOT_RESTAURANT, FORMATTED_EXAMPLE_TEMPLATE_1, NEW_EXAMPLE_JSON_TEMPLATE, few_shot_example_names, review_text, zero_shot=False)
        elif (category_name == "movie"):
            prompt = create_prompt("data/inputs/examples.json", PROMPT_TEMPLATE_FEW_SHOT_MOVIE, FORMATTED_EXAMPLE_TEMPLATE_1, NEW_EXAMPLE_JSON_TEMPLATE, few_shot_example_names, review_text, zero_shot=False)
        else:
            print("Stopped Processing: Category name must be either 'restaurant' or 'movie'")
            return
        print(prompt)
        
        new_reviews.append({"name": review_name, "text": review_text})
        print("processing review #", count_processed)
        response = get_gpt_completion(prompt)
        # print(response)
        responses_json["responses"].append(response)
        count_processed += 1 
        # Save every 25 because it might break in the middle and we don't want to lose it
        if count_processed % 25 == 0:
            now = datetime.now()      
            now1 = now.strftime("%Y_%m_%d")
            with open(f"data/outputs/{category_name}_interim_{count_processed}_gpt_raw_output_{now1}.json", "w") as raw_output_file:
                json.dump(responses_json, raw_output_file, indent=4)
                
    
    #save raw outputs to a file in data/outputs as a checkpoint
    now = datetime.now()
    now = now.strftime("%Y_%m_%d-%H_%M_%S")
    with open(f"data/outputs/{category_name}_gpt_raw_output_{now}.json", "w") as raw_output_file:
        json.dump(responses_json, raw_output_file, indent=4)

    gpt_save_output_json(new_reviews, responses_json["responses"], category_name)



if __name__ == '__main__':
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load .env using the absolute path
    dotenv_path = os.path.join(script_dir, 'environment.env')
    load_dotenv(dotenv_path)

    # UNCOMMENT THIS WHEN YOU IMPORT NEW DATA AND NEED TO PREPROCESS IT
    # preprocess("data/inputs/dataset_reviews.json")

    # Load few-shot examples
    few_shot_example_names_restaurant = ["example_1", "example_2"]
    few_shot_example_names_movie = ["example_5", "example_6"]

    with open("data/inputs/examples.json", "r", encoding="UTF-8") as examples_file:
        examples = json.load(examples_file)
        new_review = PANCAKES_EXAMPLE
        # print(create_prompt("data/inputs/examples.json", PROMPT_TEMPLATE_FEW_SHOT_RESTAURANT, FORMATTED_EXAMPLE_TEMPLATE_1, NEW_EXAMPLE_JSON_TEMPLATE, few_shot_examples, test_review, zero_shot=False))

    # Load new reviews to process
    print("==========================PROCESSING RESTAURANT REVIEWS==========================")
    with open("data/inputs/gradient/yelpshort_dataset_gradient.json", "r") as dataset_file:    
        dataset = json.load(dataset_file)
        run_gpt(dataset, few_shot_example_names_restaurant, "restaurant")
    
    print("==========================PROCESSING MOVIE REVIEWS==========================")
    # with open("data/inputs/gradient/movieshort_dataset_gradient.json", "r") as dataset_file:    
    #     dataset = json.load(dataset_file)
    #     run_gpt(dataset, few_shot_example_names_movie, "movie")
   