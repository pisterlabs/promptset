import os
import openai


client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYS_MSG = """
Identify individual ingredients from this list of ingredients that may not be comma-separated. Classify each ingredient into 4 classes ("mostly safe", "controversial", "not recommended", "unknown") based on its short-term and long-term effects on human health. If you do not have the data to support your claim, mark it "unknown". Return a JSON for each ingredient and its class.
"""

def get_lang_model_response(user_message):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": user_message}
        ],
        response_format={ "type": "json_object" }
    )
    return completion.choices[0].message.model_dump_json()
