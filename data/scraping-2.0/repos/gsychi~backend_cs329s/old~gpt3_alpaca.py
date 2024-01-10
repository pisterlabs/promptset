import json
import glob
import random
import openai
import re
from datetime import date, timedelta, datetime

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

#
# init_openai()
# processed_articles = []
# date_str = (datetime.today()-timedelta(days=2)).strftime('%Y-%m-%d')
# file_name = f"data/alpaca/{date_str}.json"
#
# with open(file_name) as f:
#     data = json.load(f)
#     for d in data:
#
#         stock_input = generate_blurb(d, "data/prompts/vaccine_prompt_alpaca.txt")
#         response = call_openai(stock_input)
#         d['blurb'] = stock_input
#         d['Prediction'] = response['choices'][0]['text']
#
#         processed_articles.append(d)
#
#
# file_name = f"data/alpaca_predictions/{date_str}.json"
# with open(file_name, "w") as f:
#     json.dump(processed_articles, f)