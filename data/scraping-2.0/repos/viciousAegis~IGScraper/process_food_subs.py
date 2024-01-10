import json
import random
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import csv
from dotenv import load_dotenv
import os

def process_data(path, num_ing=50):
    subsList = []
    ing_count = 0
    with open(path) as f:
        data = json.load(f)
        for item in data:
            if ing_count >= num_ing:
                break
            
            if len(item['Substitute']) == 0:
                continue
            
            subsList.append({
                'Ingredient': item['Ingredient'],
                'Description': item['Description'],
                'Substitutes': item['Substitute']
            })
            ing_count += 1

    for item in subsList:
        subs = item['Substitutes']
        substitutes = []
        for sub in subs:
            subDict = {
                "Sub_ingredient_name": sub['Sub_ingredient_name'],
                "Sub_tag": sub['Sub_tag'],
            }
            substitutes.append(subDict)
        item['Substitutes'] = substitutes
    
    return subsList

def prettyPrint(data):
    print(json.dumps(data, indent=4))

def generate_random_pairs(data, num_ing=20):
    pairs = []
    
    # randomly select num_ing ingredients such that each ingredient has at least 1 substitute
    selected_data = random.sample(data, num_ing)

    # randomly select 1 substitute
    for item in selected_data:
        subs = item['Substitutes']
        # randomly select 1 substitute
        subs = random.sample(subs, 1)
        pairs.append({
            'Ingredient': item['Ingredient'],
            'Description': item['Description'], # 'Description
            'Substitute': subs[0]['Sub_ingredient_name'],
        })
    
    return pairs

def get_prompt():
    template = '''
        I will give you a list of ingredients and their substitutes in the following format:
        Ingredient: [Ingredient]
        Description: [Description of the ingredient]
        Substitute: [A substitute for the ingredient]
        
        if the description is not available, you have to generate a description for the ingredient.
        You have to generate a social media post that contains the above information in a natural way. The post should be at least 4 sentences long. Do not use any special characters or hashtags. It should be written in plain English. Try to use synonyms and paraphrasing for substitution. Also, include a food item that can be made using the ingredient, and a short description of the food item.
        
        Example Inputs:
        Ingredient: Abondance
        Description: ah-bone-DAHNS This French raw cow's milk cheese has a subtle, nutty flavor. It's a good melting cheese.
        Substitute: Gruyere
        
        Post (output):
        Berthoud is a typical Savoyard dish from Chablais. It is made with Abondance cheese, white wine, garlic, and pepper. It is usually served with boiled potatoes and a green salad. To make it healthier, you can use Gruyere instead of Abondance.
        
        The output should strictly follow the format given above.
        
        Actual Inputs:
        Ingredient: {Ingredient}
        Description: {Description}
        Substitute: {Substitute}
        
        Generated Post (output):
    '''
    prompt = PromptTemplate(template=template, input_variables=["Ingredient", "Description", "Substitute"])
    return prompt

def load_llm_path():
    load_dotenv()
    local_path = (
        os.getenv("LLM_PATH")
    )
    print(local_path)
    return local_path

def prep_llm(local_path, prompt):
        # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

    # If you want to use a custom model add the backend parameter
    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
    llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True, temp=0.9, top_k=1)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain

def run_llm(data, llm_chain):
    outputs = []
    for item in data:
        ing = item['Ingredient']
        desc = item['Description']
        sub = item['Substitute']
        
        # if desc is empty, say "Generate a description for the ingredient"
        if len(desc) == 0:
            desc = "No description available."
        
        output = llm_chain.run(
            Ingredient=ing,
            Description=desc,
            Substitute=sub
        )
        print()
        print('='*50)
        outputs.append({
            'Ingredient': ing,
            'Description': desc,
            'Substitute': sub,
            'Post': output
        })
    return outputs

def save_output(outputs):
    with open('data/output.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Ingredient', 'Description', 'Substitute', 'Post'])
        writer.writeheader()
        for item in outputs:
            writer.writerow(item)

if __name__ == '__main__':
    path = './data/Ing_Sub7_FoodSub.json'
    num_ing = 50
    data = process_data(path, num_ing=num_ing*4)
    pairs = generate_random_pairs(data, num_ing=num_ing)
    
    local_path = load_llm_path()
    prompt = get_prompt()
    llm_chain = prep_llm(local_path, prompt)
    out = run_llm(pairs, llm_chain)
    save_output(out)