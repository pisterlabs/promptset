import os
import time
import json
import random
import string
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Generate the dictionary
random_dict = {
    'FastTiger': [25, 47, 51, 62, 29],
    'FastLaptop': [52, 24, 84, 85, 74],
    'FastCar': [78, 76, 94, 75, 75],
    'FastTree': [48, 70, 6, 89, 70],
    'FastBook': [35, 24, 27, 11, 51],
    'BrightTiger': [41, 29, 12, 28, 95],
    'BrightLaptop': [22, 63, 58, 92, 78],
    'BrightCar': [98, 81, 64, 32, 10],
    'BrightTree': [74, 66, 28, 54, 58],
    'BrightBook': [18, 12, 66, 39, 30],
    'ShinyTiger': [12, 16, 42, 75, 11],
    'ShinyLaptop': [27, 63, 96, 33, 59],
    'ShinyCar': [47, 8, 100, 72, 95],
    'ShinyTree': [26, 5, 25, 95, 83],
    'ShinyBook': [76, 45, 89, 89, 74],
    'SharpTiger': [25, 83, 93, 13, 3],
    'SharpLaptop': [49, 59, 58, 58, 58],
    'SharpCar': [66, 51, 90, 10, 57],
    'SharpTree': [89, 76, 42, 78, 62],
    'SharpBook': [70, 91, 47, 51, 83],
    'SmartTiger': [38, 11, 84, 74, 61],
    'SmartLaptop': [24, 84, 3, 90, 20],
    'SmartCar': [82, 53, 6, 71, 2],
    'SmartTree': [86, 94, 35, 40, 23],
    'SmartBook': [95, 7, 42, 25, 68]
}

# Initialize the ChatOpenAI model with GPT-3.5-turbo
llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define your custom prompt template for concise summaries
human_prompt_template = """Calculate the sum of values for given key {key}:

Key To the list of values:

{data}

Format the output as given format below:
sum: value_of_sum

Output:"""
human_prompt = PromptTemplate(template=human_prompt_template, input_variables=["key", "data"])

# Create an LLMChain with the chat model and custom prompt mapping
human_llm_chain = LLMChain(llm=llm, prompt=human_prompt, verbose=False)

# Convert dictionary to a human-readable format
def convert_human(data_dict):
    human_readable_str = ""
    for key, values in data_dict.items():
        values_str = ', '.join(map(str, values))
        human_readable_str += f"{key}: {values_str}\n"
    return human_readable_str

# Parse the LLM response from human-readable format
def parse_human(response):
    # Assuming the response is in the format "sum: value"
    try:
        # Extract the numerical part after "sum:"
        sum_str = response.split("sum:")[1].strip()
        return {"sum": int(sum_str)}
    except IndexError:
        raise ValueError("Response format is incorrect.")

# Function to ask GPT to calculate the sum of values for a given key
def calculate_human_sum(key):
    if key not in random_dict:
        return {"success": False, "message": f"Key '{key}' not found in the dictionary."}
    
    values = random_dict[key]
    expected_sum = sum(values)

    human_input = convert_human(random_dict)
    
    start_time = time.time()
    response = human_llm_chain.run({"key": key, "data": human_input})
    end_time = time.time()

    try:
        response_json = parse_human(response)
        calculated_sum = response_json.get('sum', None)
        
        success = calculated_sum == expected_sum
        duration = end_time - start_time

        return {"success": success, "duration": duration, "calculated_sum": calculated_sum, "expected_sum": expected_sum}

    except Exception as e:
        return {"success": False, "message": "Invalid JSON response", "output": response}


def perform_calculations(num_iterations, sleep_duration):
    total_duration = 0
    total_relative_error = 0

    count = 0
    for _ in range(num_iterations):
        for key_to_query in random_dict.keys():
            result = calculate_human_sum(key_to_query)
            
            print(f"Result for Key '{key_to_query}': {result}")

            if result.get("calculated_sum"):
                total_duration += result.get("duration", 0)
                actual_sum = result.get("expected_sum", 0)
                if actual_sum != 0:
                    relative_error = abs(result.get("calculated_sum", 0) - actual_sum) / actual_sum
                    total_relative_error += relative_error
                    count += 1

            time.sleep(sleep_duration)  # Sleep for the specified duration

    average_duration = total_duration / count if count > 0 else 0
    average_relative_error = total_relative_error / count if count > 0 else 0

    return average_duration, average_relative_error

# Example usage
num_iterations = 1  # Number of times to run the calculation
sleep_duration = 5  # Seconds to sleep between each calculation

avg_duration, avg_relative_error = perform_calculations(num_iterations, sleep_duration)
print(f"Average Duration: {avg_duration} seconds")
print(f"Average Relative Mean Error: {avg_relative_error * 100}%")


