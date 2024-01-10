import os
import json
from search import *
from prompts import COMPARE_RESUME
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import openai
import os
# from dotenv import load_dotenv
# load_dotenv()

# openai.api_key = os.environ.get('OPENAI_API_KEY')

# def gpt3_3turbo_completion(messages, summarymodel='gpt-4'): # 'gpt-3.5-turbo' or 'gtp-4'

# 	temperature = 0.2
# 	max_length = 500
# 	top_p = 1.0
# 	frequency_penalty = 0.0
# 	presence_penalty = 0.1

# 	# making API request and error checking
# 	print("making API request")
# 	try:
# 		response = openai.ChatCompletion.create(
# 			model=summarymodel, 
# 			messages=messages, 
# 			temperature=temperature, 
# 			max_tokens=max_length, 
# 			top_p=top_p, 
# 			frequency_penalty=frequency_penalty, 
# 			presence_penalty=presence_penalty)
# 	except openai.error.RateLimitError as e:
# 		print("OpenAI API rate limit error! See below:")
# 		print(e)
# 		return None, None, None
# 	except Exception as e:
# 		print("Unknown OpenAI API error! See below:")
# 		print(e)
# 		return None, None, None
	
# 	return response['choices'][0]['message']['content']

if __name__ == '__main__':
    # Load data
    with open('data/data.json', 'r') as f:
        data = json.load(f)
    # Load embeddings
    with open('data/embeddings.json', 'r') as f:
        embeddings = json.load(f)
    user_info_edu = '''Brown University Providence, RI | Sept 2017 - May 2020
Bachelor of Arts in Computer Science and Economics, GPA: 3.8/4.0
Relevant Courses: Entrepreneurial Process, Deep Learning, Statistics, Computer Systems, Algorithms, Corp Finance'''
    user_info_exp = '''Microsoft, Product Manager, Azure Media Security Redmond, WA | July 2020 – Jan 2022
• Product Manager on media protection technology PlayReady built into 5+ billion devices that secures premium
media content from the leading studios and content providers on Windows, Xbox, and 3rd party devices
• Defined a 3-year product vision focusing on tighter licensee integration, deprecation opportunities, and new
market segments. Drove annual revenue growth by 1.2x (FY20-21) to $62M in the first year
• Built, tested, and launched 4 new features in PlayReady’s new version by cross-collaborating with external
partners, developers, data scientists and marketing to close customer security gaps and enable new user scenarios
• Drove a 1-year Operations plan improving Diagnostics, Monitoring, Alerting, and Incident Management
practices for our services which reduced incident detection and mitigation time by 70%'''
    results = vector_search(user_info_exp, embeddings, data)[0:3]
    fs = []
    for result in results:
        key = result["key"]
        key_split = key.split("_")
        index = int(key_split[0])
        edu_or_exp = key_split[1]
        edu_or_exp_index = int(key_split[2])
        edu = data[f'{index}']["edu_chunks"]
        edu_text = '\n'.join(edu)
        fs_temp_0 = data[f'{index}'][f"{edu_or_exp}_chunks"][edu_or_exp_index]
        fs_temp_1 = data[f'{index}'][f"{edu_or_exp}_chunks_mod"][edu_or_exp_index]
        timeline, content = fs_temp_0.split("Content:")
        content = "Content:" + content
        fs_temp = f'''Example:
Education: {edu_text}
Entry:
{timeline}
Explanation: {fs_temp_1}
{content}'''
        fs.append(fs_temp)
        fs_prompt = "\n".join(fs)
        prompt = COMPARE_RESUME.replace("<<FS>>", fs_prompt)
        prompt = prompt.replace("<<USER_INFO>>", user_info_edu)
        prompt = prompt.replace("<<USER_EDU>>", user_info_exp)
        with open('data/final_prompt.txt', 'w') as f:
            f.write(prompt)