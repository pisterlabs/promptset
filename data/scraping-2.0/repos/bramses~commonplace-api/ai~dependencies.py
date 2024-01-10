import openai
import os
import dotenv
import tiktoken

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4")


dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

def cost_estimator(num_tokens, model_price):
    return f"DRY RUN ESTIMATED COST: ~$ {str(num_tokens * model_price) } USD"
    
gpt_costs = {
    "gpt-3.5-turbo": 0.0015 / 1000, # price per 1K tokens
    "gpt-4": 0.006 / 1000,
}

async def chat_completion(prompt, model="gpt-3.5-turbo", dry_run=False):
    if dry_run:
        return cost_estimator(len(enc.encode(prompt)), gpt_costs[model])
    completion = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ]
    )
    
    return (completion.choices[0].message.content)

async def image_completion(prompt):
 
    completion = openai.Image.create(
        prompt=prompt,
    )

    return (completion.data[0].url)

async def embed_text(text):
    completion = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
        )


    return (completion.data[0].embedding)

def send_messages(messages):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        stop=["Client:"]
    )

def arr_to_system_message(arr):
    # convert array of strings to array of objects with role: system and content: string
    return [{"role": "system", "content": x} for x in arr]
        