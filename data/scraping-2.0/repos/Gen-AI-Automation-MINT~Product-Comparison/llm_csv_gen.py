from openai import OpenAI
import json

# Initialize the OpenAI client
api_key = "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"
client = OpenAI(api_key=api_key)

# System prompt
system_prompt = """
You will be provided with semistructured data of an ecommerce product, and your task is to extract only these attributes - [title, brand, model, size, color, pack/count, material] from it and generate JSON output.
"""


# Function to process a single product and extract attributes
def process_product(product_data):
    user_prompt = f"Here is the product details: \n{json.dumps(product_data, indent=2)}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


# Load your JSON data
with open('amazon/amazon_data.json', 'r') as file:
    product_details = json.load(file)

# Process each product and collect the results
results = []
for product in product_details:
    result = process_product(product)
    results.append(result)

# Save the results to a JSON file
with open('output.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Processing complete. Data saved to output.json.")
