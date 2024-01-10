import json
import os
import traceback
import openai
from src.data_generation.utils import create_prompt, BUYER_CATEGORIES, MONTH_NAMES
from src.elastic_search.elastic_search_curd import post_data
from dotenv import load_dotenv

load_dotenv()


def generate_data(city_name):
    """
    Create data according to buyer category and month.
    """
    try:
        openai.api_key = os.environ["OPENAI_KEY"]

        generated_data = {"city_name": city_name, "buyer_categories": {}}
        for buyer_category in BUYER_CATEGORIES:
            buyer_data = {"buyer_category": buyer_category, "months": {}}
            for month_name in MONTH_NAMES:
                prompt = create_prompt(city_name, buyer_category, month_name)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=prompt,
                    max_tokens=1500,
                )
                generated_text = response["choices"][0]["message"]["content"]
                final_response = generated_text.replace("\n", "").strip()
                print(final_response)
                buyer_data["months"][month_name] = {"generated_text": json.loads(final_response)}

            # Save the buyer_data in the generated_data dictionary
            generated_data["buyer_categories"][buyer_category] = buyer_data
        post_data(generated_data)   # add data into elastic search
        return "Data has been seeded into elastic search", 200

    except Exception as e:
        traceback_str = traceback.format_exc()
        error_message = f"Exception occurred: {str(e)}\nTraceback:\n{traceback_str}"
        return error_message, 500
