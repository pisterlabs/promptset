import openai

# This chapter has a fine tuning example, which I am going to copy
# but I am not going to run it. I don't want to pay for it, so
# there may be typos or other bugs that I didn't find.

def chat_completion(prompt, model="gpt-4", temperature=0):
    res = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print(res["choices"][0]["messages"]["content"])
    
# Normally you would have your own data that you are going to train a model on
# In this case we need to generate sample data, so we can see how the fine-tuning
# works in the next section.

def build_syntetic_data():

    l_sector = ['Grocery Stores', 'Restaurants', 'Clothing Stores', 
                'Electronics Stores', 'Furniture Stores', 'Hardware Stores', 
                'Jewelry Stores', 'Pet Stores', 'Sporting Goods Stores', 'Toy Stores']
    l_city = ['London', 'Paris', 'Berlin']
    l_size = ['small', 'medium', 'large']

    f_prompt = """
    Role: You are an expert content writer with etensive skills in direct marketing
    experience. You have strong writing skills, creativity, adaptablity to 
    different tones and styles, and a deep understanding of audience needs and
    preferences for effective direct campaigns.
    Context: You have to write a short message in no more than 2 sentences for a 
    direct marketing campaign to sell a new e-comerce payment service to stores.
    The target stores have the following three characteristics:
    - The sector of activity: {sector}
    - The city where the stores are located: {city}
    - The size of the stores: {size}
    Task: Write a short message for the direct marketing campaign. Use the skills
    defined in your role to write this message! It is important that the message 
    you create takes into account the product you are selling and the
    characteristics of the store you are writing to.
    """

    f_sub_prompt = "{sector}, {city}, {size}"

    df = pd.DataFrame()
    for sector in l_sector:
        for city in l_city:
            for size in l_size:
                for i in range(3): ## 3 times each
                    prompt = f_prompt.format(sector=sector, city=city, size=size)
                    sub_prompt = f_sub_prompt.format(sector=sector, city=city, size=size)
                    response_txt = chat_completion(
                        prompt, model="gpt-3.5-turbo", temperature=1
                    )
                    new_row = {"prompt": sub_prompt, "completion": response_txt}
                    new_row = pd.DataFrame([new_row])
                    df = pd.concat([df, new_row], ignore_index=True, axis=0)
    df.to_csv("out_openai_completion.csv", index=False)
    
# At the command line prepare the data
# openai tools fine_tunes.prepare_data -f out_openai_completion.csv 

def upload_finetining():
    ft_file = openai.File.create(
        file=open("out_openai_completion_prepared.jsonl", "rb"), purpose="fine-tune"
    )    
    openai.FineTune.create(
        training_fil=ft_file["id"], model="davinci", suffix="direct_marketning"
    )
    
# Not sure why they have a code example and a command line example but they do
# the fact that copilot knew the command line example suggests it is needed
# openai tools fine_tunes.create -m davinci -s direct_marketing -f out_openai_completion_prepared.jsonl
    
def use_model():
    openai.Completion.create(
        model="davinci:ft-book:direct-marketing-2023-05-01-15-20-35",
        prompt="Hotel, New York, small ->",
        max_tokens=100,
        temperature=0,
        stop="\n"
    )