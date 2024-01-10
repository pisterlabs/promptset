import openai
from dotenv import load_dotenv
load_dotenv() 
import os


openai.api_key  = str(os.getenv('OPEN_AI_API_KEY'))

# def get_completion(prompt, index,model="gpt-3.5-turbo",):
#     messages = [
#         {"role":"system","content":"You are an expert in writing personalized emails for product marketing."},
#         {"role": "user", "content": prompt}
#         ]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0.5, 
#         n=5
#     )
     
    # if(index>0 and response.choices[0].message['content'] =="Please, enter a valid product description." ):
    #     raise ValueError("Sorry, you've attempted multiple times. Please try with a different input this time.")

    # if(response.choices[0].message['content'] =="Please, enter a valid product description."):
    #     raise ValueError("Please, enter a valid product description.")
    
    # return json.loads(response.choices[index].message['content'])

def get_completion_valid(prompt, model="gpt-3.5-turbo"):
    messages = [
        {"role":"system","content":" You are an automated product and service validation expert."},
        {"role": "user", "content": prompt}
        ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        
         # this is the degree of randomness of the model's output
    )
    #return response.choices[].message["content"]
    return response.choices

def get_completion_email(prompt, model="gpt-3.5-turbo"):
    messages = [
        {"role":"system","content":"You are an expert in writing personalized emails for product marketing."},
        {"role": "user", "content": prompt}
        ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,
        n = 5
        
         # this is the degree of randomness of the model's output
    )
    #return response.choices[].message["content"]
    return response.choices
    

# Attributes = {"Product Description" : "The product is Adidas Shoes. It ","Email Tone":"Excitement", "Email Tone Description":"Generate enthusiasm for new product launches or sales."}
# temp_descp = {Attributes["Product Description"]}
# Personal_Information={'title': 'Mr', 'first_name': 'Steve', 'middle_initial': 'R', 'last_name': 'Edgerly', 'address': '8700 West Ln Spc 126', 'city': 'Stockton', 'state': 'CA', 'zipcode': 95210, 'countycode': 77, 'county': 'SAN JOAQUIN', 'age_range': '65 - 69', 'income_range': 'Under $20,000', 'potential_investor': 'Likely Investor', 'home_owner': 'Yes', 'marital_status': 'Single', 'ethnicity': 'English', 'language': 'English', 'gender': 'Male', 'estimated_household_income': 'Under $20,000', 'housing_type': 'Trailer Court', 'estimated_home_value': '$1 - $24,999', 'year_home_built': 1975, 'mail_order_purchase': 'Golf                        Car Buff                    Computer                    Health/Fitness              Books And Music             Finance/Investment          Sweepstakes/Gambling        Apparel', 'actual_age': 68, 'birth_month': 'October', 'number_of_children': 0, 'veterans': 'No', 'home_size': 0, 'health': 'General Health & Exercise, Fitness', 'motor_interest': 'Auto Racing Enthusiast', 'politics': 'Politically Liberal', 'purchase_behavior': 'Catalog Shopper', 'technology_entertainment': 'Internet User, Personal Computers, Internet Access'}



def remove_null_or_empty_values(dictionary):
    return {key: value for key, value in dictionary.items() if value is not None and value != ""}

def valid(temp_description):
    return f"""

Your main goal is to assess the validity of a product or service based on its minimal description.

 

Please provide a brief description of the product or service:

User: {temp_description}

 

Is the description valid? (True/False)

 

IMPORTANT: If the description is ambiguous or unclear, please answer "False" to indicate uncertainty.

"""

def email(Attributes,Personal_Information):
    return f"""your objective is to compose a concise, personalized \
email with subject using the provided information. 

Attributes:```{Attributes}```\

Personal Information:```{Personal_Information}```\

Follow the guidelines below:\
1. Generate a short email with maximum 2 lines in each paragraph with maximum 3 paragraphs.
2. Use the "Attributes" and "Personal Information" delimited in triple backticks for generating dynamic content in email.\
3. Attributes section is provided by Product owner and Personal information is of a target person.\
4. Compose a compelling email that aligns with the given Product Description, email tone and Email tone description under attributes.\
5. Regardless of the tone of the email, refrain from revealing any Personally Identifiable Information (PII), such as income, age, and the like.\
6. Feel free to use emojis or other Unicode characters to add emphasis or express emotions in the email.\
7. Strictly, Do not include product descriptions, features, or any special offers, deals, or discounts in the email, unless explicitly mentioned in the Product Description.\
8. Utilize as many fields under "Personal Information" as possible in combination to create an engaging \
email.\
9. If you come across any irrelevant fields under "Personal Information", unrelated to the product \
or unsuitable for the email, please omit them. Prioritize the recipient's name \
and relevant details to ensure a meaningful email. \
10. Remember you are prohibited from including PII data fields present in Personal Information under Attributes in email \
and focus on engaging the recipient with a personalized message. \
11. Generate email in json format, with "subject","regards" and each paragraph in different key like "para1", "para2",etc. \
12. Please ensure that the generated email does not contain any vulgar language or offensive content and maintains a professional and respectful tone.\
"""

def responseGenerator(personal_information,Attributes):
    
    temp_description = Attributes["Product Description"]
    personal_information = remove_null_or_empty_values(personal_information)
    del personal_information["language"]

    prompt_valid = valid(temp_description)
    prompt_email = email(Attributes, personal_information)

    response_valid=get_completion_valid(prompt_valid)

    if response_valid[0]['message']['content']=="True":
        response_email=get_completion_email(prompt_email)
        return response_email
    else:
        raise ValueError("Please, provide better and detailed description about product.") 



#     prompt = f"""your objective is to compose a concise, personalized \
# email with subject using the provided information. 
# Before generating email,
#     i. first check product description under Attributes and analyze product description.\
#     ii. Given a product description in one word, treat the input word as case insensitive, check if it corresponds to an existing product or brand name. Please only consider nouns and noun phrases as valid product descriptions and exclude adjectives, adverbs, and verbs from the response."\
#     iii. Given a product description in sentences, verify that the description pertains to a specific product and does not contain any irrelevant information beyond the product description.\
#     iv. If any of ii or iii condition is not satisfying then don't follow below guidelines and just print "Please, Enter a valid product description."\

# Follow the guidelines below:\
# 1. Generate a short email with maximum 2 lines in each paragraph with maximum 3 paragraphs.
# 2. Use the "Attributes" and "Personal Information" delimited in triple backticks for generating dynamic content in email.\
# 3. Attributes section is provided by Product owner and Personal information is of a target person.\
# 4. Compose a compelling email that aligns with the given Product Description, email tone and Email tone description under attributes.\
# 5. Feel free to use emojis or other Unicode characters to add emphasis or express emotions in the email.\
# 6. Strictly, Do not include product descriptions, features, or any special offers, deals, or discounts in the email, unless explicitly mentioned in the Product Description.\
# 7. Utilize as many fields under "Personal Information" as possible in combination to create an engaging \
# email.\
# 8. If you come across any irrelevant fields under "Personal Information", unrelated to the product \
# or unsuitable for the email, please omit them. Prioritize the recipient's name \
# and relevant details to ensure a meaningful email. \
# 9. Remember you are prohibited from including PII data fields present in Personal Information under Attributes in email \
# and focus on engaging the recipient with a personalized message. \
# 10. Generate email in json format, with "subject","regards" and each paragraph in different key like "para1", "para2",etc. 

  
# Attributes:```{Attributes}```\

# Personal Information:```{personal_information}```\

# """
    
#     return get_completion(prompt,index)
