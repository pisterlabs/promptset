import openai
import os
import json 

api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key

def example1():
    openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        
        messages=[
            {"role": "system", "content": "You are a helpful teacher."},
            {
                "role": "user",
                "content": "Are there other measures than time complexity for an \
                    algorithm?",
            },
            {"role": "assistant",
            "content": "Yes, there are other measures besides time complexity \
                for an algorithm, such as space complexity."
            },
            {"role": "user", "contet": "What is it?"},        
        ],
    )

def find_product(sql_query):
    # Execute query here
    results = [ 
               {"name": "pen", "color": "blue", "price": 1.99},
               {"name": "pen", "color": "red", "price": 1.78},
    ]
    return results 

functions = [
    {
        "name": "find_product",
        "description": "Get a list of products from a sql query",
        "parameters": {
            "type": "object",
            "properties": { 
                "sql_query": {
                    "type": "string",
                    "description": "A SQL query",
                }
            },
        "required": ["sql_query"],
        },
    }
]

# I can't really follow what this function is trying to do, so I don't know why it is giving 
# me an error. I suspect I probably need to lode some json to have things I can interact with
# hopefully the later examples will be better

def using_functions():
    user_question = "I need the top 2 products where the price is less than 2.00"
    messages = [{"role": "user", "content": user_question}]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", messages=messages, functions=functions
    )


    print(response)
    response_message = response["choices"][0]["message"]
    print(response_message)
    print(response.function_call)
    messages.append(response_message)
    
    function_args = json.loads(
        response_message["function_call"]["arguments"]
    )
    products = find_product(function_args.get("sql_query"))
    
    messages.append({
        "role": "function",
        "name": "find_product",
        "content": json.dumps(products)
    })
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", messages=messages,
    )
    
def test_embedding(text):
    result = openai.Embedding.create(
        model="text-embedding-ada-002", input=text
    )
    print(result)
    print(result['data'])
    print(type(result['data']))
    print(len(result['data'][0]['embedding'])) # 1536
    
def moderation():
    response = openai.Moderation.create(
        model="text-moderation-latest",
        input="I want to kill my neighbor."
    )    
    # per the book, this gets flagged for violence
    # And then it shows scores, hate was 0.04, violence was .94, all others were e-06 or less
    
    
    
    
if __name__ == "__main__":
    # example1()
    # using_functions()
    test_embedding("I like to eat pizza")