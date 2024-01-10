import sys
sys.path.append('/Users/pranav/anaconda3/lib/python3.11/site-packages')
sys.path.append('//Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')

from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from apikey import APIKEY2
import os
import json
import re

os.environ["GOOGLE_API_KEY"] = APIKEY2

def seoStats(url):

    main_template = PromptTemplate(
        input_variables = ['url'],
        template = 'Give me the web traffic, PageRank, and the amount of backlinks it has for the following url {url}. Use B for any values in billion and use M for any values in million, Make sure to provide the values in a JSON dictionary.'
    )

    model = GooglePalm(temperature=0.1)

    title_chain = LLMChain(llm = model, prompt = main_template, verbose = False, output_key = 'details')

    ans = title_chain.run(url)

    start_index = ans.find('{')
    end_index = ans.rfind('}') + 1

    # Extract the JSON dictionary
    json_data = ans[start_index:end_index]

    # Define a regular expression pattern to match numbers with optional decimal points and "B" suffix
    pattern = r'([\d.]+)([BM]?)'

    # Function to replace matched values with numeric equivalents
    def replace(match):
        value, suffix = match.groups()
        if suffix.lower() == 'b':
            return str(int(float(value) * 1e9))
        elif suffix.lower() == 'm':
            return str(int(float(value) * 1e6))
        else:
            return value

    # Use the regular expression pattern and replace function to modify the JSON data
    try :
        json_data = re.sub(pattern, replace, json_data)

        #print(json_data)

        data = json.loads(json_data)

        # Extract the features
        web_traffic = int(data.get("web_traffic"))
        PageRank = float(data.get("pagerank"))
        backlinks = int(data.get("backlinks"))

        return web_traffic,PageRank,backlinks
    except Exception as e:
        web_traffic = 0
        PageRank = 0
        backlinks = 0
        return web_traffic,PageRank,backlinks