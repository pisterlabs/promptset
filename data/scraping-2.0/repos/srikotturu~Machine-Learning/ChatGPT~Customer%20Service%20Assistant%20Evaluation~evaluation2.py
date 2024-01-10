import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
import utils

# Load the API key from a .env file
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['API_KEY']

# Define a function to get completions from messages
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

# Define a function to evaluate the assistant's answer with a rubric
def evaluate_with_rubric(customer_msg, product_info, assistant_answer):
    system_message = "You are an assistant that evaluates how well the customer service agent answers a user question by looking at the context that the customer service agent is using to generate its response."

    user_message = f"""You are evaluating a submitted answer to a question based on the context that the agent uses to answer the question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {customer_msg}
    ************
    [Context]: {product_info}
    ************
    [Submission]: {assistant_answer}
    ************
    [END DATA]

Compare the factual content of the submitted answer with the context. Ignore any differences in style, grammar, or punctuation.
Answer the following questions:
    - Is the Assistant response based only on the context provided? (Y or N)
    - Does the answer include information that is not provided in the context? (Y or N)
    - Is there any disagreement between the response and the context? (Y or N)
    - Count how many questions the user asked. (output a number)
    - For each question that the user asked, is there a corresponding answer to it?
      Question 1: (Y or N)
      Question 2: (Y or N)
      ...
      Question N: (Y or N)
    - Of the number of questions asked, how many of these questions were addressed by the answer? (output a number)
"""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = get_completion_from_messages(messages)
    return response

# Define a function to evaluate the assistant's answer compared to the ideal answer
def evaluate_vs_ideal(customer_msg, ideal_answer, assistant_answer):
    system_message = "You are an assistant that evaluates how well the customer service agent answers a user question by comparing the response to the ideal (expert) response. Output a single letter and nothing else."

    user_message = f"""You are comparing a submitted answer to an expert answer on a given question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {customer_msg}
    ************
    [Expert]: {ideal_answer}
    ************
    [Submission]: {assistant_answer}
    ************
    [END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
    (A) The submitted answer is a subset of the expert answer and is fully consistent with it.
    (B) The submitted answer is a superset of the expert answer and is fully consistent with it.
    (C) The submitted answer contains all the same details as the expert answer.
    (D) There is a disagreement between the submitted answer and the expert answer.
    (E) The answers differ, but these differences don't matter from the perspective of factuality.
  choice_strings: ABCDE
"""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = get_completion_from_messages(messages)
    return response

# Main function to perform all tasks
def main():
    # Customer message with product-related queries
    customer_msg = "tell me about the smartx pro phone and the fotosnap camera, the dslr one. Also, what TVs or TV related products do you have?"

    # Get products based on the customer's message
    products_by_category = utils.get_products_from_query(customer_msg)
    category_and_product_list = utils.read_string_to_list(products_by_category)
    product_info = utils.get_mentioned_product_info(category_and_product_list)

    # Generate an answer using the provided information
    assistant_answer = utils.answer_user_msg(user_msg=customer_msg, product_info=product_info)

    # Print the assistant's answer
    print(f"Assistant's Answer for customer message:\n{assistant_answer}")

    # Evaluate the assistant's answer using the rubric
    evaluation_output = evaluate_with_rubric(customer_msg, product_info, assistant_answer)
    print(f"Evaluation Output:\n{evaluation_output}")

    # Define a test set with the customer message and an ideal answer
    test_set_ideal = {
        'customer_msg': "tell me about the smartx pro phone and the fotosnap camera, the dslr one. Also, what TVs or TV related products do you have",
        'ideal_answer': "Of course! The SmartX Pro Phone is a powerful smartphone with advanced camera features. For instance, it has a 12MP dual camera. Other features include 5G wireless and 128GB storage. It also has a 6.1-inch display. The price is $899.99.\n\nThe FotoSnap DSLR Camera is great for capturing stunning photos and videos. Some features include 1080p video, 3-inch LCD, a 24.2MP sensor, and interchangeable lenses. The price is $599.99.\n\nFor TVs and TV-related products, we offer 3 TVs...\n\nAll TVs offer HDR and Smart TV. The CineView 4K TV has vibrant colors and smart features. Some of these features include a 55-inch display, 4K resolution. It's priced at $599. The CineView 8K TV is a stunning 8K TV. Some features include a 65-inch display and 8K resolution. It's priced at $2999.99. The CineView OLED TV lets you experience vibrant colors. Some features include a 55-inch display and 4K resolution. It's priced at $1499.99.\n\nWe also offer 2 home theater products, both of which include Bluetooth. The SoundMax Home Theater is a powerful home theater system for an immersive audio experience. Its features include 5.1 channel, 1000W output, and a wireless subwoofer. It's priced at $399.99. The SoundMax Soundbar is a sleek and powerful soundbar. Its features include 2.1 channel, 300W output, and a wireless subwoofer. It's priced at $199.99\n\nAre there any additional questions you may have about these products that you mentioned here? Or do you have other questions I can help you with?"
    }

    # Evaluate the assistant's answer compared to the ideal answer
    evaluation_result = evaluate_vs_ideal(test_set_ideal['customer_msg'], test_set_ideal['ideal_answer'], assistant_answer)
    print(f"Evaluation result 1:\n{evaluation_result}")

    # A different assistant answer for comparison
    assistant_answer_2 = "life is like a box of chocolates"

    # Evaluate the second assistant answer compared to the ideal answer
    evaluation_result_2 = evaluate_vs_ideal(test_set_ideal['customer_msg'], test_set_ideal['ideal_answer'], assistant_answer_2)
    print(f"Evaluation result 2:\n{evaluation_result_2}")

if __name__ == "__main__":
    main()
