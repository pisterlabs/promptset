import requests
import openai
openai.api_key = "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"


headers = {'Authorization': 'Bearer b7d97a8e-a112-4893-bd98-2b3fe5a9e1e5', 'user-agent': 'GPT_Test'}


def pid(x):
    url = 'https://api.squarespace.com/1.0/commerce/products/'
    updated_url = url + x
    r = requests.get(updated_url, headers=headers)
    return r.text


obj = pid('640f6d7046bdd0207616116f')


def generate_product_description():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "This JSON object represents a product on my website. Can you write me a short product description that suggests ways the product can be used? "},
            {"role": "user", "content": obj}
                  ],
        max_tokens=256,
        n=1,
        stop=None,
        temperature=0.7,
    )
    # description = response.choices[0]
    description = response['choices'][0]['message']['content']
    return description


product_description = generate_product_description()
print(product_description)
