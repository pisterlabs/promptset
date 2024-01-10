from sqlite_module import * # to interact with database
import os # to access environment variables
from openai import OpenAI


def chatbot_completion(prompt, model="gpt-3.5-turbo-1106"): # gpt-4-1106-preview
    """use ChatGPT to answer a prompt"""
    try:   
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(model=model,
        messages=[
            # {"role": "system", "content": ""},
            # {"role": "user", "content": ""},
            # {"role": "assistant", "content": ""},
            {"role": "system", "content": "You are a helpful assistant that will explain why two people's expectations are similar to each others, and why they might make a good pair (friend/couple/soul mate or whatever). You will give five reasons why so, and format them in bullet points."},
            {"role": "user", "content": f"{prompt}."},
        ],
        temperature=0)
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # usage
    user_id = 'jimmy#C@gmail.com'
    database_path = 'user.db'
    user_exp = db_data_read(user_id, 'expectation', database_path)
    user_match_result =  db_data_read(user_id,'match_result_id', database_path) # matched id and score, enclosed in list
    for i in user_match_result:
        matched_id, matched_score = i[0], i[1]
        matched_name, matched_exp = db_data_read(matched_id, 'name', database_path), db_data_read(matched_id, 'expectation', database_path)
        prompt = f"My expectation is '{user_exp}', and his/her expectation is '{matched_exp}'. Why are we a good match?"
        result = chatbot_completion(prompt)
        
        print(f"You are matched with {matched_name} with a score of {matched_score}!")
        print(f"Your expectation is: {user_exp}")
        print(f"{matched_name}'s expectation is: {matched_exp}")
        print(result)
    
if __name__ == '__main__':
    main()