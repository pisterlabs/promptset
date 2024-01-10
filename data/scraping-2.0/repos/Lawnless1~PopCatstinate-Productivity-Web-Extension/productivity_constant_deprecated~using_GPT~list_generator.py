import os
import openai
import json

openai.api_key = "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"

script_dir = os.path.dirname(__file__)
read_filepath = os.path.join(script_dir, 'top_300_websites.txt')
write_filepath = os.path.join(script_dir, "top_300_website_productivity_score.txt")
with open(read_filepath, 'r') as file:
    contents = file.read()
    
    



list_contents = contents.replace(" ", "")
list_contents = contents.split("\n")

domain_list = []
for website in list_contents:
     base_domain = website.split('.')[0]
     if base_domain not in domain_list:
        domain_list += [base_domain]

prompted_website = ""
websites_in_thing = 0
total_website_list_with_scores = ""

for website in domain_list[242:]:
    prompted_website += website
    prompted_website += ", " 
    websites_in_thing += 1
    if websites_in_thing > 4:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are concise assistant"},
                {"role": "user", "content": "For these 5 websites, you will assign a score between 0.00 and 1.00, about the likeliness of being productive on that website. 0.00 is the least productive and 1.00 is the most productive. You will not give a single answer that is a multiple of 0.05. Give your answer in csv format, naming the website and its score for each row. You will only give your csv answer. You will not say anything else. You won't include a legend. You will not number the list. So don't write 1. google, 0.6 2. facebook, 0.3, instead write google, 0.6 facebook 0.4. These are the 5 websites {}".format(prompted_website)}
            ],
			temperature = 0
        )
        json_content = json.loads(str(completion))
        websites_output = json_content['choices'][0]['message']['content']
        total_website_list_with_scores += "\n" + websites_output
        websites_in_thing = 0
        prompted_website = ""
	
with open(write_filepath, "a+") as file:
	file.write(total_website_list_with_scores )
 


