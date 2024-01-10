import openai
import sys
sys.path.append('..')
import utils
sys.path.append('..')
import json

delimiter = "####"

# Input Moderation
def test_Moderation(comment):
    response = openai.Moderation.create(comment)
    moderation_output = response["results"][0]
    if moderation_output["flagged"] != False:
        return "The response is not appropriate!"
    else:
        return "The response is appropriate!"

# Prevent Prompt Injection
def test_Prompt_Injection(user_Input, language):
    system_message = f"""
    Assistant responses must be in English or {language}. \
    If the user says something in other languages, \
    always respond in English. The user input \
    message will be delimited with {delimiter} characters.
    """
    user_message_for_model = f"""User message, \
    remember that your response to the user \
    must be in English or {language}: \
    {delimiter}{user_Input}{delimiter}
    """

    messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': user_message_for_model},  
    ] 
    response = utils.get_completion_from_messages(messages)
    print(response)
    

# Classificaiton of Service Requests
def get_Classification_of_Service_Request(user_Message):
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with \
    {delimiter} characters.
    Classify each query into a primary category \
    and a secondary category. 
    Provide your output in json format with the \
    keys: primary and secondary.

    Primary categories: Billing, Technical Support, \
    Account Management, or General Inquiry.

    Billing secondary categories:
    Unsubscribe or upgrade
    Add a payment method
    Explanation for charge
    Dispute a charge

    Technical Support secondary categories:
    General troubleshooting
    Device compatibility
    Software updates

    Account Management secondary categories:
    Password reset
    Update personal information
    Close account
    Account security

    General Inquiry secondary categories:
    Product information
    Pricing
    Feedback
    Speak to a human

    """
    
    messages =  [  
    {'role':'system', 
    'content': system_message},    
    {'role':'user', 
    'content': f"{delimiter}{user_Message}{delimiter}"},  
    ] 
    response = utils.get_completion_from_messages(messages)
    return response

#  Answering user questions using Chain of Thought Reasoning
def chain_of_thought_reasoning(user_message):
    system_message = f"""
    Follow these steps to answer the customer queries.
    The customer query will be delimited with four hashtags,\
    i.e. {delimiter}. 

    Step 1:{delimiter} First decide whether the user is \
    asking a question about a specific product or products. \
    Product cateogry doesn't count. 

    Step 2:{delimiter} If the user is asking about \
    specific products, identify whether \
    the products are in the following list.
    All available products: 
    1. Product: TechPro Ultrabook
    Category: Computers and Laptops
    Brand: TechPro
    Model Number: TP-UB100
    Warranty: 1 year
    Rating: 4.5
    Features: 13.3-inch display, 8GB RAM, 256GB SSD, Intel Core i5 processor
    Description: A sleek and lightweight ultrabook for everyday use.
    Price: $799.99

    2. Product: BlueWave Gaming Laptop
    Category: Computers and Laptops
    Brand: BlueWave
    Model Number: BW-GL200
    Warranty: 2 years
    Rating: 4.7
    Features: 15.6-inch display, 16GB RAM, 512GB SSD, NVIDIA GeForce RTX 3060
    Description: A high-performance gaming laptop for an immersive experience.
    Price: $1199.99

    3. Product: PowerLite Convertible
    Category: Computers and Laptops
    Brand: PowerLite
    Model Number: PL-CV300
    Warranty: 1 year
    Rating: 4.3
    Features: 14-inch touchscreen, 8GB RAM, 256GB SSD, 360-degree hinge
    Description: A versatile convertible laptop with a responsive touchscreen.
    Price: $699.99

    4. Product: TechPro Desktop
    Category: Computers and Laptops
    Brand: TechPro
    Model Number: TP-DT500
    Warranty: 1 year
    Rating: 4.4
    Features: Intel Core i7 processor, 16GB RAM, 1TB HDD, NVIDIA GeForce GTX 1660
    Description: A powerful desktop computer for work and play.
    Price: $999.99

    5. Product: BlueWave Chromebook
    Category: Computers and Laptops
    Brand: BlueWave
    Model Number: BW-CB100
    Warranty: 1 year
    Rating: 4.1
    Features: 11.6-inch display, 4GB RAM, 32GB eMMC, Chrome OS
    Description: A compact and affordable Chromebook for everyday tasks.
    Price: $249.99

    Step 3:{delimiter} If the message contains products \
    in the list above, list any assumptions that the \
    user is making in their \
    message e.g. that Laptop X is bigger than \
    Laptop Y, or that Laptop Z has a 2 year warranty.

    Step 4:{delimiter}: If the user made any assumptions, \
    figure out whether the assumption is true based on your \
    product information. 

    Step 5:{delimiter}: First, politely correct the \
    customer's incorrect assumptions if applicable. \
    Only mention or reference products in the list of \
    5 available products, as these are the only 5 \
    products that the store sells. \
    Answer the customer in a friendly tone.

    Use the following format:
    Step 1:{delimiter} <step 1 reasoning>
    Step 2:{delimiter} <step 2 reasoning>
    Step 3:{delimiter} <step 3 reasoning>
    Step 4:{delimiter} <step 4 reasoning>
    Response to user:{delimiter} <response to customer>

    Make sure to include {delimiter} to separate every step.
    """
    messages =  [  
    {'role':'system', 
    'content': system_message},    
    {'role':'user', 
    'content': f"{delimiter}{user_message}{delimiter}"},  
    ] 

    response = utils.get_completion_from_messages(messages)
    return response

# Check Output using model self-evaluate
def check_Output_self_evaluate(customer_message, final_response_to_customer):
    system_message = f"""
    You are an assistant that evaluates whether \
    customer service agent responses sufficiently \
    answer customer questions, and also validates that \
    all the facts the assistant cites from the product \
    information are correct.
    The product information and user and customer \
    service agent messages will be delimited by \
    3 backticks, i.e. ```.
    Respond with a Y or N character, with no punctuation:
    Y - if the output sufficiently answers the question \
    AND the response correctly uses product information
    N - otherwise

    Output a single letter only.
    """
    product_information = """{ "name": "SmartX ProPhone", "category": "Smartphones and Accessories", "brand": "SmartX", "model_number": "SX-PP10", "warranty": "1 year", "rating": 4.6, "features": [ "6.1-inch display", "128GB storage", "12MP dual camera", "5G" ], "description": "A powerful smartphone with advanced camera features.", "price": 899.99 } { "name": "FotoSnap DSLR Camera", "category": "Cameras and Camcorders", "brand": "FotoSnap", "model_number": "FS-DSLR200", "warranty": "1 year", "rating": 4.7, "features": [ "24.2MP sensor", "1080p video", "3-inch LCD", "Interchangeable lenses" ], "description": "Capture stunning photos and videos with this versatile DSLR camera.", "price": 599.99 } { "name": "CineView 4K TV", "category": "Televisions and Home Theater Systems", "brand": "CineView", "model_number": "CV-4K55", "warranty": "2 years", "rating": 4.8, "features": [ "55-inch display", "4K resolution", "HDR", "Smart TV" ], "description": "A stunning 4K TV with vibrant colors and smart features.", "price": 599.99 } { "name": "SoundMax Home Theater", "category": "Televisions and Home Theater Systems", "brand": "SoundMax", "model_number": "SM-HT100", "warranty": "1 year", "rating": 4.4, "features": [ "5.1 channel", "1000W output", "Wireless subwoofer", "Bluetooth" ], "description": "A powerful home theater system for an immersive audio experience.", "price": 399.99 } { "name": "CineView 8K TV", "category": "Televisions and Home Theater Systems", "brand": "CineView", "model_number": "CV-8K65", "warranty": "2 years", "rating": 4.9, "features": [ "65-inch display", "8K resolution", "HDR", "Smart TV" ], "description": "Experience the future of television with this stunning 8K TV.", "price": 2999.99 } { "name": "SoundMax Soundbar", "category": "Televisions and Home Theater Systems", "brand": "SoundMax", "model_number": "SM-SB50", "warranty": "1 year", "rating": 4.3, "features": [ "2.1 channel", "300W output", "Wireless subwoofer", "Bluetooth" ], "description": "Upgrade your TV's audio with this sleek and powerful soundbar.", "price": 199.99 } { "name": "CineView OLED TV", "category": "Televisions and Home Theater Systems", "brand": "CineView", "model_number": "CV-OLED55", "warranty": "2 years", "rating": 4.7, "features": [ "55-inch display", "4K resolution", "HDR", "Smart TV" ], "description": "Experience true blacks and vibrant colors with this OLED TV.", "price": 1499.99 }"""

    q_a_pair = f"""
    Customer message: ```{customer_message}```
    Product information: ```{product_information}```
    Agent response: ```{final_response_to_customer}```

    Does the response use the retrieved information correctly?
    Does the response sufficiently answer the question

    Output Y or N
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': q_a_pair}
    ]

    response = utils.get_completion_from_messages(messages, max_tokens=1)
    return response

def find_category_and_product_v1(user_input,products_and_category):

    delimiter = "####"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    AND
        'products': <a list of products that must be found in the allowed products below>


    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.
    

    List out all products that are relevant to the customer service query based on how closely it relates
    to the product name and product category.
    Do not assume, from the name of the product, any features or attributes such as relative quality or price.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {products_and_category}
    

    """
    
    few_shot_user_1 = """I want the most expensive computer."""
    few_shot_assistant_1 = """ 
    [{'category': 'Computers and Laptops', \
    'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """
    
    messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},  
    {'role':'assistant', 'content': few_shot_assistant_1 },
    {'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"},  
    ] 
    return utils.get_completion_from_messages(messages)


def find_category_and_product_v2(user_input,products_and_category):
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    AND
        'products': <a list of products that must be found in the allowed products below>
    Do not output any additional text that is not in JSON format.
    Do not write any explanatory text after outputting the requested JSON.


    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.
    

    List out all products that are relevant to the customer service query based on how closely it relates
    to the product name and product category.
    Do not assume, from the name of the product, any features or attributes such as relative quality or price.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {products_and_category}
    

    """
    
    few_shot_user_1 = """I want the most expensive computer. What do you recommend?"""
    few_shot_assistant_1 = """ 
    [{'category': 'Computers and Laptops', \
    'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """
    
    few_shot_user_2 = """I want the most cheapest computer. What do you recommend?"""
    few_shot_assistant_2 = """ 
    [{'category': 'Computers and Laptops', \
    'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """
    
    messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},  
    {'role':'assistant', 'content': few_shot_assistant_1 },
    {'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"},  
    {'role':'assistant', 'content': few_shot_assistant_2 },
    {'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"},  
    ] 
    return utils.get_completion_from_messages(messages)

# Evaluate Response with ideal answers
import json
def eval_response_with_ideal(response,ideal,debug=False):
    if debug:
        print("response")
        print(response)
    
    # json.loads() expects double quotes, not single quotes
    json_like_str = response.replace("'",'"')
    
    # parse into a list of dictionaries
    l_of_d = json.loads(json_like_str)
    
    # special case when response is empty list
    if l_of_d == [] and ideal == []:
        return 1
    
    # otherwise, response is empty 
    # or ideal should be empty, there's a mismatch
    elif l_of_d == [] or ideal == []:
        return 0
    
    correct = 0    
    
    if debug:
        print("l_of_d is")
        print(l_of_d)
    for d in l_of_d:

        cat = d.get('category')
        prod_l = d.get('products')
        if cat and prod_l:
            # convert list to set for comparison
            prod_set = set(prod_l)
            # get ideal set of products
            ideal_cat = ideal.get(cat)
            if ideal_cat:
                prod_set_ideal = set(ideal.get(cat))
            else:
                if debug:
                    print(f"did not find category {cat} in ideal")
                    print(f"ideal: {ideal}")
                continue
                
            if debug:
                print("prod_set\n",prod_set)
                print()
                print("prod_set_ideal\n",prod_set_ideal)

            if prod_set == prod_set_ideal:
                if debug:
                    print("correct")
                correct +=1
            else:
                print("incorrect")
                print(f"prod_set: {prod_set}")
                print(f"prod_set_ideal: {prod_set_ideal}")
                if prod_set <= prod_set_ideal:
                    print("response is a subset of the ideal answer")
                elif prod_set >= prod_set_ideal:
                    print("response is a superset of the ideal answer")

    # count correct over total number of items in list
    pc_correct = correct / len(l_of_d)
        
    return pc_correct

def evaluate_all_pair_set(msg_ideal_pairs_set):
    # Note, this will not work if any of the api calls time out
    score_accum = 0
    for i, pair in enumerate(msg_ideal_pairs_set):
        print(f"example {i}")
        
        customer_msg = pair['customer_msg']
        ideal = pair['ideal_answer']
        
        # print("Customer message",customer_msg)
        # print("ideal:",ideal)
        response = find_category_and_product_v2(customer_msg, utils.get_products_and_category())

        
        # print("products_by_category",products_by_category)
        score = eval_response_with_ideal(response,ideal,debug=False)
        print(f"{i}: {score}")
        score_accum += score   

    n_examples = len(msg_ideal_pairs_set)
    fraction_correct = score_accum / n_examples
    print(f"Fraction correct out of {n_examples}: {fraction_correct}")
  
# Evaluate with rubric  
def eval_with_rubric(test_set, assistant_answer):
    cust_msg = test_set['customer_msg']
    context = test_set['context']
    completion = assistant_answer
    
    system_message = """\
    You are an assistant that evaluates how well the customer service agent \
    answers a user question by looking at the context that the customer service \
    agent is using to generate its response. 
    """

    user_message = f"""\
    You are evaluating a submitted answer to a question based on the context \
    that the agent uses to answer the question.
    Here is the data:
        [BEGIN DATA]
        ************
        [Question]: {cust_msg}
        ************
        [Context]: {context}
        ************
        [Submission]: {completion}
        ************
        [END DATA]

    Compare the factual content of the submitted answer with the context. \
    Ignore any differences in style, grammar, or punctuation.
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

    response = utils.get_completion_from_messages(messages)
    return response

def eval_vs_ideal(test_set, assistant_answer):

    cust_msg = test_set['customer_msg']
    ideal = test_set['ideal_answer']
    completion = assistant_answer
    
    system_message = """\
    You are an assistant that evaluates how well the customer service agent \
    answers a user question by comparing the response to the ideal (expert) response
    Output a single letter and nothing else. 
    """

    user_message = f"""\
    You are comparing a submitted answer to an expert answer on a given question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {cust_msg}
    ************
    [Expert]: {ideal}
    ************
    [Submission]: {completion}
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

    response = utils.get_completion_from_messages(messages)
    return response