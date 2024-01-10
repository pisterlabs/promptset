# requirements_file --------------------

!pip install -U openai

# function_import --------------------

import openai

# function_code --------------------

def generate_slogan(api_key, ecom_website_name):
    """
    Generate a slogan for an e-commerce website selling eco-friendly products using GPT-3.

    :param api_key: str, the API key provided by OpenAI.
    :param ecom_website_name: str, the name of the e-commerce website.
    :return: str, the generated slogan.
    """
    openai.api_key = api_key
    prompt = f"Generate a catchy slogan for {ecom_website_name} that sells eco-friendly products."
    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt,
        max_tokens=6,
        n=1,
        temperature=0.7
    )
    slogan = response.choices[0].text.strip()
    return slogan

# test_function_code --------------------

def test_generate_slogan():
    print("Testing generate_slogan function.")
    api_key = "your_api_key_here"  # replace with your actual API key
    ecom_website_name = "EcoShop"
    
    slogan = generate_slogan(api_key, ecom_website_name)
    print(f"Generated slogan: {slogan}")
    
    assert len(slogan) > 0, "Failed to generate a slogan."
    
    print("Test passed.")

# Run the test function
test_generate_slogan()