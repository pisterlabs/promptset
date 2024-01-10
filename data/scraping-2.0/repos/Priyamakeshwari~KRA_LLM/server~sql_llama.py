# from langchain.utilities import SQLDatabase
# from langchain.llms import LlamaCpp
# from langchain_experimental.sql import SQLDatabaseChain

# db = SQLDatabase.from_uri("sqlite:///db/data_KRA.sqlite")  # Updated database URL
# llm = LlamaCpp(
#         model_path='/Users/kuldeep/Project/KRA_LLM/server/models/llama-2-13b-chat.Q5_K_M.gguf',
#         max_tokens=4096,
#         temperature=0.9,
#         top_p=1,
#         verbose=False,
#         n_batch=512,
#         n_gpu_layers=40,
#         f16_kv=True, 
#         streaming=True,
#     )

# print(db)
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
# print(db_chain)
# # db_chain.run("How many employees are there in employees table?")
# # db_chain.run('select employee database and display all columns in the table')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-llama-2-7B")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-llama-2-7B", torch_dtype=torch.bfloat16)

text = """CREATE TABLE stadium (
    stadium_id number,
    location text,
    name text,
    capacity number,
    highest number,
    lowest number,
    average number
)

CREATE TABLE singer (
    singer_id number,
    name text,
    country text,
    song_name text,
    song_release_year text,
    age number,
    is_male others
)

CREATE TABLE concert (
    concert_id number,
    concert_name text,
    theme text,
    stadium_id text,
    year text
)

CREATE TABLE singer_in_concert (
    concert_id number,
    singer_id text
)

-- Using valid SQLite, answer the following questions for the tables provided above.

-- What is the maximum, the average, and the minimum capacity of stadiums ?

SELECT"""

input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=500)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
