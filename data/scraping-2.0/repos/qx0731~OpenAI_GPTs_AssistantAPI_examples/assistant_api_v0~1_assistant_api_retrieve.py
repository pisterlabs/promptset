# something 
import os
import time
import random
from faker import Faker
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['OPENAI_API_KEY']

## step0: generate a fake file as the knowledge to upload 
# fake = Faker()

# # Define some sample categories
# categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books']

# # Generate data
# data = []

# for _ in range(50):
#     product_name = fake.word().title()
#     category = random.choice(categories)
#     original_price = round(random.uniform(10, 1000), 2)  # Prices between 10 and 1000
#     discount = round(random.uniform(5, 30), 2)  # Discount percentage between 5% and 30%
#     coupon = f"{discount}% OFF"

#     # Format as a TSV row
#     row = f"{product_name}\t{category}\t${original_price}\t{coupon}"
#     data.append(row)

# # Output to a TSV format (as a string)
# tsv_data = "\n".join(data)

# # Print or save to a file
# print(tsv_data)

# with open('coupons.tsv', 'w') as file:
#     file.write(tsv_data)


client = OpenAI(
    api_key =  api_key
)

# step1: upload file 
file = client.files.create(
    file = open("coupons.tsv", "rb"), 
    purpose = 'assistants'
)

print(file.id)

file_list = client.files.list()
print(file_list)

# step2: create the assistants with the file 
assistant = client.beta.assistants.update(
    assistant_id='asst_euxJG7taJ3yfXHitwd2nbeI2', 
    instructions= 'You are a chatbot designed to help find coupon information using the uploaded file', 
    model = 'gpt-4-1106-preview', 
    tools=[{"type":"retrieval"}], 
    file_ids = [file.id]
)
print(assistant.id) 

my_assistant = client.beta.assistants.list(
    order = 'desc', 
    limit = "20"
)
print(my_assistant.data)

# step3: create a thread 
thread = client.beta.threads.create(
    
)
print(thread)

# step4: add more message to the thread 
message = client.beta.threads.messages.create(
    thread_id = thread.id, 
    role = 'user', 
    content = 'can you tell me the category of product that has coupon'
) 

# step5: run the assitant to get the response 
run = client.beta.threads.runs.create(
    thread_id = thread.id, 
    assistant_id= assistant.id, 
    instructions = "please address the user as Jane, and provide the information Jane is asking for"
    
)
print(run.id)

# step6: retrieve the run status 
print(run.status)
while run.status not in ["completed", "failed"]:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id, 
        run_id = run.id
    )
    print(run.status)
    time.sleep(10)
    
# step4.2: add more message to the thread 
message = client.beta.threads.messages.create(
    thread_id = thread.id, 
    role = 'user', 
    content = 'tell me the coupon in Books'
) 

# step5.2: run the assitant to get the response 
run = client.beta.threads.runs.create(
    thread_id = thread.id, 
    assistant_id= assistant.id    
)
print(run.id)

# step6.2: retrieve the run status 
print(run.status)
while run.status not in ["completed", "failed"]:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id, 
        run_id = run.id
    )
    print(run.status)
    time.sleep(10)
    
# step7: 
messages = client.beta.threads.messages.list(
    thread_id=thread.id
) 

for m in messages: 
    print(m.role + ":" + m.content[0].text.value) 
    print("============================")
