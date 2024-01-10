import os
from pathlib import Path
import re
from . import scraping_utils, nlp_utils, general_utils
import numpy as np
import time
import yaml
from retry import retry

import openai


with open("server_src/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg = cfg["config"]

max_req_to_server = cfg["max_req_to_server"]
qa_model = cfg["QA_MODEL"]
BULLETS_GENERIC_STYLE_NAME = cfg["BULLETS_GENERIC_STYLE_NAME"]
TABULARIZE_STYLE_NAME = cfg["TABULARIZE_STYLE_NAME"]
MAX_GPT_PASSES = cfg["MAX_GPT_PASSES"]
MIN_MARKED_TEXT_LENGTH = cfg["MIN_MARKED_TEXT_LENGTH"]
MULTIPLE_PASSES_MAX_TOKENS = cfg["MULTIPLE_PASSES_MAX_TOKENS"]
CHARS_TO_DECREASE_ON_DECLINE = cfg["chars_to_decrease_on_decline"]
COMPLETION_TIMEOUT = cfg["COMPLETION_TIMEOUT"]

openai.api_key = os.getenv("OPENAI_API_KEY")
if not os.getenv("OPENAI_API_KEY"):
    # try to get the key from file in parent folder
    with open('../../openai_gpt3_key.txt', 'r') as f:
        openai.api_key = f.read()

azure_flag = False
if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
    openai.api_version = "2023-06-01-preview" # "2023-05-15"
    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_flag = True

def parse_gpt_response_bullets(response: str):
    """Converts a string of the form '- bullet1 \n- bullet2' to a [bullet1, bullet2]."""
    
    stripped_s = response.strip()

    # remove lines which are whitespace only
    stripped_s = re.sub(r'^[\s]*\n', '', stripped_s, flags=re.MULTILINE)

    l_dict = []
    for j, line in enumerate(stripped_s.split('\n')):
        if len(line) < 3: 
            continue
        line = line.strip()
        if line[1].isdigit():
            line = line[2:]
        else:
            line = line.strip()[1:]

        # find index of first letter or number in any alphabet
        first_letter_index = -1
        for i, c in enumerate(line):
            if c.isalnum():
                first_letter_index = i
                break
        if first_letter_index == -1:
            continue
        
        l_dict.append({'key': j+1, 'value': line[first_letter_index:]})

    return l_dict

def look_for_delimiter_inside_paranthesis(s, delimiter, replacement_delimiter):
    open_paranthesis_flag = False
    new_s = ''
    for c in s:
        if c == '(':
            open_paranthesis_flag = True
        elif c == ')':
            open_paranthesis_flag = False
        if c == delimiter and open_paranthesis_flag:
            new_s += replacement_delimiter
            continue
        new_s += c
    return new_s

def look_for_delimiter_inside_table(s, table_delimiter, delimiter, replacement_delimiter):
    inside_table = False
    new_s = ''
    prev = 0
    for i, c in enumerate(s):
        if c == table_delimiter:
            inside_table = True
        elif c == delimiter and inside_table:
            new_s += s[prev:i] + replacement_delimiter
            prev = i + 1
    new_s += s[prev:]
    return new_s

def look_for_delimiter_after_comma(s, delimiter, replacement_delimiter, comma=','):
    comma_occ = []
    colon_occ = []
    for i, c in enumerate(s):
        if c == comma:
            comma_occ.append(i)
        elif c == delimiter:
            colon_occ.append(i)

    replace_colon = []
    for colon_acc_i in colon_occ[::-1]:
        while len(comma_occ) > 0 and comma_occ[-1] > colon_acc_i:
            comma_occ.pop()
        if len(comma_occ) == 0:
            break
        replace_colon.append(comma_occ[-1])

    new_s = ''
    prev = 0
    for replace_colon_i in replace_colon:
        new_s += s[prev:replace_colon_i].strip() + replacement_delimiter
        prev = replace_colon_i + 1
    new_s += s[prev:].strip()
    return new_s

def is_likely_key_val_list(s):
    """Returns True if the string is likely to be of the form 'key1:\nvalue1\nvalue2\nkey2:value1'"""
    if len(s) < 10:
        return False
    if s.count(':') < 2:
        return False
    if s.count('\n') < 2:
        return False
    s = s.strip()
    s_lines = s.split('\n')
    if s_lines[0].count(':') == 0:
        return False
    if s_lines[0].split(':')[1].strip() != '':
        return False
    return True

def parse_gpt_response(s, style='', category_delimiter=':', table_delimiter='|', special_time_token='<<STT>>', special_cat_token='<<CAT>>', special_https_token='<<|HTTP|>>', values_to_drop=['', '-', 'None', 'N/A', 'n/a', 'N/a', 'NA', 'Not available', 'Not Available', 'not available', 'Not available.', 'Not Available.', 'not available.', 'varies', 'Varies', 'Unknown', 'unknown', 'Not provided', 'not mentioned', 'none mentioned']):
    """Converts a string, usually of the form 'key1:value1,key2:value2', to a dictionary/JSON."""
    
    if style == BULLETS_GENERIC_STYLE_NAME:
        if not is_likely_key_val_list(s):
            if s.strip()[:s.find('\n')].count(':') == 1:
                s = '- ' + s.strip()
            print('Parsing as bullet points.')
            parsed_s = parse_gpt_response_bullets(s)
            return parsed_s
        # remove incoming '-'. It's a thing that ChatGPT does.
        s = '\n'.join([line if not line.startswith('- ') else line[2:] for line in s.splitlines()])
    
    if style == TABULARIZE_STYLE_NAME:
        if not s.startswith('Table:'):
            s = 'Table:\n' + s

    stripped_s = s.strip()
    
    # replace places in stripped_s in which a digit occurs before and after ':' with special_time_token
    stripped_s = re.sub(r'(\d+)(:)(\d+)', r'\1'+special_time_token+r'\3', stripped_s)
    
    # replace places where 'http://' or 'https://' occurs with special_https_token
    url_pattern = re.compile(r'(https?://[^\s:]+(?::\d+)?(?:/[^\s]*)?)', re.IGNORECASE)
    stripped_s = url_pattern.sub(lambda match: match.group().replace(":", special_https_token), stripped_s)
    
    # check if string is not a list
    if category_delimiter not in stripped_s[0:30]: # If the string does not contain ':' in the first ~30 characters, then it is probably not a list.
        return [{'key': ' ', 'value': stripped_s}]

    # proceed to parsing of the form {key1: value1, key2: value2, ...}, with values potentially being tables
    s_lines = stripped_s.split('\n')
    n_lines = len(s_lines)
    
    # if s contains category_delimiter (initially ':'), then it is usually a key-value pair (until next line with ':')
    # EXCPETIONS: ':' could appear elsewhere, e.g. in paranthesis or if the model failed to start a new line
    # we improve it now.
    new_s_lines = []
    for s_lines_i in s_lines:
        # case 1: check if ':' is in paranthesis, i.e. it has one '(' some characters before and one ')' some characeters after it.
        new_s_line = look_for_delimiter_inside_paranthesis(s_lines_i, category_delimiter, special_cat_token)
        # case 1a: check if ':' is in a table, i.e. it has one '|' some characters before it.
        new_s_line = look_for_delimiter_inside_table(new_s_line, table_delimiter, category_delimiter, special_cat_token)
        # case 2: a ',' that comes before a ':' is probably a mistake - need to replace ',' with '\n'
        new_s_line = look_for_delimiter_after_comma(new_s_line, category_delimiter, '\n')
        
        new_s_lines.append(new_s_line)
    s_lines = new_s_lines

    d = {}
    # iterate over the list of strings
    last_key = '' # last key - used to store multi-line values
    for i in range(n_lines):
        # split the string into parts separated by ':'
        s_split_i = s_lines[i].split(category_delimiter)
        
        # if the string contains category_delimiter - e.g., ':'
        if len(s_split_i) > 1:
            last_key = ''
            # add the key-value pair to the dictionary
            d[s_split_i[0].strip()] = s_split_i[1].strip()
            if s_split_i[1].strip() == '':
                last_key = s_split_i[0].strip()
                d[last_key] = []
        elif last_key != '':
            # if the string does not contain ':', then it should be a table
            # count number of '|' in string s
            n_pipes = s_lines[i].count(table_delimiter)
            # if s contains '|', then it is a table
            if n_pipes > 0:
                # split the string into parts separated by '|'
                s_split_i = s_lines[i].split(table_delimiter)
                # if the string contains '|'
                if len(s_split_i) > 1:
                    # add the key-value pair to the dictionary
                    d[last_key].append(s_lines[i])
            else: # if s does not contain '|', then it is a multi-line value. For now we treat it the same
                d[last_key].append(s_lines[i])

    # recursively run on d, apply .replace(special_time_token, ':') on all strings
    # also apply .replace(special_cat_token, ':'/category_delimiter) on all strings
    for key in d:
        if isinstance(d[key], list):
            for i in range(len(d[key])):
                d[key][i] = d[key][i].replace(special_time_token, category_delimiter)
                d[key][i] = d[key][i].replace(special_cat_token, category_delimiter)
                d[key][i] = d[key][i].replace(special_https_token, category_delimiter)
        else:
            d[key] = d[key].replace(special_time_token, category_delimiter)
            d[key] = d[key].replace(special_cat_token, category_delimiter)
            d[key] = d[key].replace(special_https_token, ':')
    
    '''
    # split comma-separated values - currently unused.
    for key in d:
        if isinstance(d[key], list): 
            d[key] = d[key]
            continue
        # print(d[key].split(','))
        d[key] = [s_i.strip() for s_i in d[key].split(',')]
    '''

    # "plaster" - replace empty lists by empty strings
    for key in d:
        if d[key] == []:
            d[key] = ''

    # split "tables" into lists
    for key in d:
        if isinstance(d[key], list):
            prev_col_len = 0 # we keep track of the number of columns in the previous row. Sometimes the table is not aligned, and we need to add a column to the beginning of the row.
            for i in range(len(d[key])):
                columns = d[key][i].count(table_delimiter)
                if columns < prev_col_len:
                    d[key][i] = [s_i.strip() for s_i in (' ' + table_delimiter + ' ' + d[key][i]).split(table_delimiter)]
                else:
                    d[key][i] = [s_i.strip() for s_i in d[key][i].split(table_delimiter)]
                    prev_col_len = columns

    # remove "empty" values - i.e. values that are not lists and are in values_to_drop
    new_d = {}
    for key in d:
        if not isinstance(d[key], list):
            if d[key] not in values_to_drop:
                # remove key from d
                # .pop(key, None)
                new_d[key] = d[key]
        else:
            new_d[key] = d[key]
    d = new_d

    # convert {key: val} to [{key: key, value: val}]
    l_dict = []
    for key in d:
        l_dict.append({'key': key, 'value': d[key]})

    return l_dict

def get_gpt_prompt(style='travel'):
    if style == 'travel':
        prompt_title = "You help a traveler design a multi-day or multi-destination itinerary and gather information about a trip. Convert the blog entries to structured data. When writing a table, put different destinations or activities in separate rows.\n"
        input_prompt = "Text: "
        output_prompt = "\n\nOutput, possible fields {Activity name, Accommodation, Eating, Transportation, Best Seasons, Preparation, Budget, Itinerary table}:\n" # "Structured data:\nActivity:"
        example_pairs = [
            ["We had an amazing time in Peru, especially in Huaraz. We went to the a bunch of day treks and the Santa Cruz trek! It is a 3-day trek in the Andes mountains. In the first day, we walked 4 hours on a rugged trail and camped near a river. In the second day, we had tough 16 kilometers through beautiful terrain. In the third day, we went over a high altitude pass of 5100m, finishing the trek in a small town. We rode on a shared taxi back to Huaraz. The whole thing cost us about 400 Soles.\n\n",
            '''Activity name: Santa Cruz trek
Accommodation: camping
Transportation: shared taxi
Budget: 400 Soles
Itinerary table:
Day | Length | Details
1 | 4 hrs | rugged, river camping
2 | 16 km | beautiful terrain
3 |  | 5100m pass'''],
            ["Recommended hotels in France, where mid/high end means over 100 euros per night, budget means less.\n\n",
             '''Destination name: France
Hotels:
Location | Name | Details
Paris | Hotel de Crillon | High-end
 | Hotel de Ville | Mid-range
Lyon | Comte Ornon | High-range
 | Hotel Boutique | Mid-range
Bourdeaux | Hotel de Seze | Mid-range
 | Best Western Francais | Budget'''],
            ["In the last summer I was in Nepal, exploring the Himalayas. In one of the most memorable experiences, I went on the Tilicho lake trek. After hiring a porter at Pokhara, I took a bus to Besi-Sahar. After a day of altitude acclimatization in Manang, enjoying the local food at a simple hotel, I set out at sunrise to Tilicho lake base camp. This day was beautiful but a little dangerous, as some paths suffer from landslide. After another night at a simple hotel, I began the climb to the lake. After about 3 hours and 1000m of climb, I made it to the lake. Boy-oh-boy, the views were amazing! Snow-capped mountains with a far-reaching pristine lake in the middle. The walk was definitely worth it. After climbing down I stopped at base camp for another night with a hearty meal along fellow travelers. In the next day, I hiked back 15 km to Manang and made the trip back to Pokhara.",
            '''Activity name: Tilicho lake trek
Accommodation: simple hotels
Transportation: bus to Besi-Sahar
Itinerary table:
Day | Destination | Details
1 | Manang | altitude acclimatization
2 | Base camp | landslide danger
3 | Tilicho lake and back | 3 hours, 1000m climb
4 | Manang | 15km hike
| Pokhara | ''']
        ]
        keywords = ['Santa Cruz', 'Andes', 'Huaraz', '5100m', 'Crillon', 'France', 'Paris', 'Lyon', 'Comte', 'Ornon', 'Western', 'Bourdeaux', 'Seze', 'Tilicho', 'Pokhara', 'Manang', 'Besi-Sahar']
        
        continued_prompt_title = "You help a traveler design a multi-day or multi-destination itinerary and gather information about a trip. You are given the data collected so far and a relevant body of text. You need to use the text to add details and expand the data and output the revised data in the same format. Be informative and succinct.\n"
        continued_prev_data_prompt = "\n\nPrevious data:\n"
        continued_new_text_prompt = "\n\nNew text:\n"
        continued_output_prompt = "\n\nRevised data:\n"
        continued_prompt_dict = {
            "continued_prompt_title": continued_prompt_title, 
            "continued_prev_data_prompt": continued_prev_data_prompt, 
            "continued_new_text_prompt": continued_new_text_prompt, 
            "continued_output_prompt": continued_output_prompt,
            "keywords": ["multi-day", "gather information about a trip", "Be informative and succinct"]}
    elif style == 'bizanalytics':
        prompt_title = "You are trying to help an analyst appraise businesses and gather information from business news. Convert the following text snippets to structured data.\n"
        input_prompt = "Text: "
        output_prompt = "\n\nOutput, possible fields include {Main company/ies, Business/service, Valuation, Product, Features, Pricing, Investors, Business decisions/events, Area, Personnel, Challenges}:\n"
        example_pairs = [
            ['''On-demand shuttle and software company Via has raised another $130 million, capital that has pushed its valuation to about $3.3 billion as demand from cities to update its legacy transit systems rises.
The round was led by Janus Henderson with participation from funds and accounts managed by BlackRock, ION Crossover Partners, Koch Disruptive Technologies and existing investor Exor. To date, the company has raised $800 million.
Via, which today employs about 950 people, has two sides to its business. The company operates consumer-facing shuttles in Washington, D.C. and New York. Its underlying software platform, which it sells to cities, transportation authorities, school districts and universities to deploy their own shuttles, is not only the core of its business; it has become the primary driver of growth.
Co-founder and CEO Daniel Ramot previously told TechCrunch that there was was little interest from cities in the software-as-a-service platform when the company first launched in 2012. Via landed its first city partnership with Austin in late 2017, after providing the platform to the transit authority for free. It was enough to allow Via to develop case studies and convince other cities to buy into the service. In 2019, the partnerships side of the business “took off,” Ramot said in an interview last year.
Today, the software side — branded internally as TransitTech — has eclipsed its consumer-facing operations. Via said TransitTech revenue more than doubled year on year to exceed an annual run rate of $100 million. The software platform is used by more than 500 partners, including Los Angeles Metro. Jersey City and Miami in the United States as well as Arriva Bus UK, a Deutsche Bahn company that uses it for a first and last-mile service connecting commuters to a high-speed train station in Kent, U.K.
Via doesn’t provide specifics on what it plans to use the funds for. The company has made two acquisitions in the past 18 months, including Fleetonomy in 2020.
Earlier this year, Via used $100 million in cash and equity to acquire a company called RemixCorpTM, a startup that developed mapping software used by cities for transportation planning and street design. The startup became a subsidiary of Via, an arrangement that will let the startup maintain its independent brand.\n\n''',
            '''Main company: Via
Business/service: On-demand shuttle, software-as-a-service
Since: 2012
Total funding: $800M
Valuation: $3.3B
Revenue: Doubling YOY
Investors: Janus Henderson, BlackRock, ION Crossover Partners, Koch Disruptive Technologies, Exor
Geography: Washington, D.C., New York, Austin, Los Angeles Metro, Jersey City, Miami, Arriva Bus UK
Clients: over 500
Personnel:
Daniel Ramot | CEO
Employees | 950
Business decisions:
Type | Details
Funding round | $130M
Acquired Fleetonomy | 2020
Acquired RemixCorpTM | $100M, cash and equity, mapping software'''
            ],
        ]
        keywords = ['Via', 'Daniel Ramot', 'Fleetonomy', 'RemixCorpTM', 'Los Angeles Metro', 'Arriva Bus UK','Deutsche Bahn', 'Janus Henderson', 'BlackRock', 'ION Crossover Partners', 'Koch Disruptive Technologies', 'Exor']
        continued_prompt_dict = None
    elif style == 'spaper':
        prompt_title = "You are trying to help an academic researcher to quickly understand the key points of a scientific paper. In the following, convert each text snippet to structured data.\n"
        input_prompt = "Text: "
        output_prompt = "\n\nOutput, possible fields {Scientific field, Background, Novelty, Conclusions/Key takeaways, Methods}:\n"
        example_pairs = [
            ['''Ultra-diffuse galaxies that contain a large sample of globular clusters (GCs) offer an opportunity to test the predictions of galactic dynamics theory. NGC5846-UDG1 is an excellent example, with a high-quality sample of dozens of GC candidates. We show that the observed distribution of GCs in NGC5846-UDG1 is suggestive of mass segregation induced by gravitational dynamical friction. We present simple analytic calculations, backed by a series of numerical simulations, that naturally explain the observed present-day pattern of GC masses and radial positions. Subject to some assumptions on the GC population at birth, the analysis supports the possibility that NGC5846-UDG1 resides in a massive dark matter halo. This is an example for the use of GC-rich systems as dynamical (in addition to kinematical) tracers of dark matter.\n\n''' ,
            '''Scientific field: Galaxies, globular clusters, dark matter
Background: Ultra-diffuse galaxies that contain a large sample of globular clusters (GCs) offer an opportunity to test the predictions of galactic dynamics theory. NGC5846-UDG1 is an excellent example, with a high-quality sample of dozens of GC candidates.
Novelty: NGC5846-UDG1 has a high-quality sample of dozens of GC candidates and dynamical friction is likely effective in the galaxy
Main conclusion: NGC5846-UDG1 is an example for the use of GC-rich systems as dynamical (in addition to kinematical) tracers of dark matter
Methods: simple analytic calculations, numerical simulations''']]
        keywords = ['NGC5846-UDG1', 'galaxies', 'globular clusters', 'dark matter', 'dynamical friction']
        continued_prompt_dict = None
    elif style == 'spaper_variant':
        prompt_title = "You are trying to help an academic researcher to quickly understand the key points of a scientific paper. In the following, convert each text snippet to structured data.\n"
        input_prompt = "Text: "
        output_prompt = "\n\nOutput, possible fields {Scientific field, Background, Novelty, Conclusions/Key takeaways, Methods}:\n"
        example_pairs = [
            ['''The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. We implement sequence ordering by using fixed positional encodings. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.\n\n''' ,
            '''Scientific field: Neural networks, machine translation
Background: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism.
Novelty: We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
Key achievements: Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task and 41.8 on the WMT 2014 English-to-French translation task, establishing a new single-model state-of-the-art BLEU score. The model generalizes well to other tasks and can be trained relatively fast.
Methods: Attention mechanism, fixed positional encodings, performance on language tasks''']]
        keywords = ['attention mechanism', 'Attention mechanism', 'BLEU', 'German', 'neural', 'convolutions', 'Transformer', 'positional encodings']
        continued_prompt_dict = None
    elif style == 'generic':
        prompt_title = "You are trying to help a layperson get a summary with the main background required to understand the following text and the main conclusions that stem from it. The summary should not exceed 8 sentences.\n"
        input_prompt = "Text: "
        output_prompt = "\n\nSummary:\n"
        example_pairs = []
        keywords = ['exceed 8 sentences']
        continued_prompt_dict = None
    elif style == BULLETS_GENERIC_STYLE_NAME:
        prompt_title = "Summarize the following text into bullet points. Try to make the bullet points progress in logic, i.e. background would appear before conclusions. Be informative and succinct.\n"
        input_prompt = "Text: "
        output_prompt = "\n\nBullet points:\n"
        example_pairs = []
        keywords = ['into bullet points', 'progress in logic']
        
        continued_prompt_title = "You help a user get the essence of a body of text. You are given the bullet points collected so far and a relevant body of text. You need to use the text to add details and expand the bullet points and output the revised bullet points. Try to make the bullet points progress in logic, i.e. background would appear before conclusions. Be informative and succinct. If the new text does not appear to be relevant, you can ignore it and output the previous bullet points."
        continued_prev_data_prompt = "\n\nPrevious bullet points:\n"
        continued_new_text_prompt = "\n\nNew text:\n"
        continued_output_prompt = "\n\nRevised bullet points:\n"
        continued_prompt_dict = {
            "continued_prompt_title": continued_prompt_title, 
            "continued_prev_data_prompt": continued_prev_data_prompt, 
            "continued_new_text_prompt": continued_new_text_prompt, 
            "continued_output_prompt": continued_output_prompt,
            "keywords": ["get the essence of a body", "You are given the bullet points", "Be informative and succinct"]}
    elif style == 'criticizepaper':    
        prompt_title = "You are helping a reviewer review a scientific paper. You are given an excerpt from a paper with the purpose of finding flaws in logic, execution, etc. Summarize your report in bullet points. Try to support your criticism with quotes from the text. If you can't find flaws, do not say any.\n"
        input_prompt = "Paper excerpt: "
        output_prompt = "\n\nCritical review of flaws in the paper:\n"
        example_pairs = []
        keywords = []
        
        continued_prompt_dict = None
    elif style == 'explain':
        prompt_title = 'You are helping someone read complicated text. Given some text, do your best to explain the text in simple terms. Do not drop key aspects of the text.'
        input_prompt = 'Text: '
        output_prompt = '\n\nExplanation:\n'
        example_pairs = []
        keywords = []

        continued_prompt_dict = None
    elif style == 'tabularize':
        prompt_title = 'You are helping parse textual data into a table. The table cells should be separated by \'|\' and new lines.'
        input_prompt = 'Text: '
        output_prompt = '\n\nTable:\n'
        example_pairs = [
            ['''Above limb I(195.12 Å)
(arcsec) erg cm2s−1sr−1
I-P P
0.00 52.45 164.67
1.00 62.02 235.34
2.00 69.19 338.49
3.00 75.52 466.16\n\n''' ,
'''Above limb | I(195.12 Å) |
(arcsec) | erg cm2s−1sr−1 |
 | I-P | P
0.00 | 52.45 | 164.67
1.00 | 62.02 | 235.34
2.00 | 69.19 | 338.49
3.00 | 75.52 | 466.16
''']]

        keywords = ['parse textual data', 'table cells should be separated by \'|\'']

        continued_prompt_dict = None

    else:
        raise ValueError(f"style {style} not supported")

    examples = [input_prompt + example_pair[0] + output_prompt + example_pair[1] for example_pair in example_pairs]

    return prompt_title, input_prompt, output_prompt, examples, example_pairs, keywords, continued_prompt_dict

def hijack_and_bad_quality_check(coarse_text: str, response_text: str, keywords: list):
    '''
    if any of the keywords is not in the coarse_text but is in the response_text, 
    then the response_text is hijacked or copied from the examples and should be discarded.

    Returns '' if the query is hijacked or copied from the examples, otherwise returns the response_text
    '''
    for keyword in keywords:
        if keyword in response_text and keyword not in coarse_text:
            print('SEEMS TO BE HIJACKED OR COPIED FROM THE EXAMPLES')
            print(response_text)
            return True
    return False

def gpt_response_to_clean_text(response, model):
    response_text = ''
    if model == 'text-davinci-003' or model == 'text-curie-001':    
        response_text = response['choices'][0].text
        response_text = re.sub(r'\n{2,}', '\n', response_text)
        response_text = response_text.strip()
    elif model == "gpt-3.5-turbo" or model == "gpt-4":
        response_text = response["choices"][-1]["message"]["content"]
        response_text = re.sub(r'\n{2,}', '\n', response_text)
        response_text = response_text.strip()
    return response_text

@retry(exceptions=openai.error.Timeout, tries=4)
def gpt_completion(query_to_model, max_tokens=768, model='text-davinci-003', prev_msgs=[]):
    if model == 'text-davinci-003' or model == 'text-curie-001':
        print('Operating on ' + model)
        return openai.Completion.create(
            model=model, 
            prompt=query_to_model,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            request_timeout=COMPLETION_TIMEOUT)
    elif model == "gpt-3.5-turbo" or model == "gpt-4":
        print('Operating on ' + model)
        new_msgs = prev_msgs + [{"role": "user", "content": query_to_model}] if query_to_model != '' else prev_msgs
        if azure_flag:
            print('Using Azure completion')
            return openai.ChatCompletion.create(
                engine="essence-gpt35turbo", 
                messages=new_msgs)
        else:
            return openai.ChatCompletion.create(
                model=model, 
                messages=new_msgs,
                request_timeout=COMPLETION_TIMEOUT)
    return None

def get_gpt_response(prompt, coarse_text, output_prompt, keywords, model, initial_char_index=0, final_char_index=10000, max_tokens=768):
    final_char_index = min(final_char_index, len(coarse_text))

    successful_response = False
    number_of_attemps = 0
    while not successful_response and number_of_attemps < max_req_to_server and final_char_index > initial_char_index:
        number_of_attemps += 1

        text_to_decode = coarse_text[initial_char_index:final_char_index] # limit the text, since models are limited to 4k tokens
        query_to_model = prompt + text_to_decode + output_prompt
        try:
            response = gpt_completion(query_to_model, max_tokens=max_tokens, model=model)
            response_text = gpt_response_to_clean_text(response, model)
            successful_response = True
        except openai.error.Timeout:
            time.sleep(0.3)
            print('BIG Timeout. Trying again.')
            continue
        except Exception as e:
            print(e)
            final_char_index -= CHARS_TO_DECREASE_ON_DECLINE
            print('Decreasing amount of tokens. New final_char_index: ', final_char_index)
            time.sleep(0.3) # wait a little, to not to overquery the API
            
    if number_of_attemps == max_req_to_server:
        return False, 'ERROR: Server encountered problems or query is long.', 0
    
    if final_char_index < initial_char_index:
        return False, 'ERROR: Processing failed.', 0
    
    if (keywords is not None) and hijack_and_bad_quality_check(coarse_text, response_text, keywords):
        raise Exception('Hijacked')
    
    return True, response_text, final_char_index

def get_gpt_summary(coarse_text, style='travel', max_char_length=10000, model='text-davinci-003', backwards_chars=0):
    """Get a summary based on GPT-3 API.
    coarse_text: text to be parsed by GPT-3
    style: style of the text, e.g. 'travel'
    final_char_index: index of the last character of coarse_text to be processed by GPT-3

    The prompts are defined in get_gpt_prompt(style) function.
    """
    try:
        prompt_title, input_prompt, output_prompt, examples, example_pairs, keywords, continued_prompt_dict = get_gpt_prompt(style=style)
    except ValueError as e:
        print(e)
        return '', 0

    gpt_credits = 1

    # single pass
    print('First pass... ' + str(len(coarse_text)))
    success_flag, response_text, actual_final_char_index = get_gpt_response(
        prompt_title + ''.join(examples) + input_prompt, 
        coarse_text, 
        output_prompt, 
        keywords,
        model,
        initial_char_index=0, final_char_index=max_char_length)

    # multiple passes
    if (continued_prompt_dict is not None) and success_flag and (actual_final_char_index < len(coarse_text)):
        gpt_credits += 1
        passes = 1
        while success_flag and (actual_final_char_index < len(coarse_text)) and passes < MAX_GPT_PASSES:
            print('Continuing with the next pass...')
            success_flag, response_text, actual_final_char_index = get_gpt_response(
                continued_prompt_dict["continued_prompt_title"] + continued_prompt_dict["continued_prev_data_prompt"] + response_text + continued_prompt_dict["continued_new_text_prompt"], 
                coarse_text, 
                continued_prompt_dict['continued_output_prompt'], 
                continued_prompt_dict['keywords'], 
                model, 
                initial_char_index=actual_final_char_index - backwards_chars, final_char_index=actual_final_char_index + max_char_length - backwards_chars,
                max_tokens=MULTIPLE_PASSES_MAX_TOKENS)
            passes += 1

    return response_text, gpt_credits

def get_title_for_entry(coarse_text, query_to_model='', model='gpt-3.5-turbo') -> str:
    """
    Get a title of the entry from the text.
    coarse_text: text to be processed by a title-generating-model
    model: model to be used by OpenAI API (currently we only use OpenAI, but other models can be used)
    """

    # We prompt the model with the very beginning of the text, assuming that the title is there
    # We do not supply the model with any examples, to remain agnostic to the style
    if query_to_model == '':
        query_to_model = "Summarize this text to something that can serve as a title that labels the text.\nText:\n" + coarse_text[0:300] + "\nTitle:"

    successful_flag = False
    number_of_attempts = 0
    while (not successful_flag) and number_of_attempts < 5:
        try:
            response = gpt_completion(query_to_model, max_tokens=64, model=model)
            successful_flag = True
            response_text = gpt_response_to_clean_text(response, model)
        except Exception as e:
            print(e)
            number_of_attempts += 1
            time.sleep(0.5)
    if not successful_flag:
        response_text = 'ERROR OCCURRED'

    # some specific cleanings for title generation
    response_text = response_text.replace('"', '') # remove quotes, since they sometimes come up as wrapping of the title in the output
    if response_text[-1] == '.': 
        response_text = response_text[:-1]  # remove '.' in the end of response_text if exists

    return response_text

def process_url(request_dict, data_path, max_char_length=1000, model='text-davinci-003'):
    """Process URL and return structured data.
    request_dict: dictionary with the following keys:
        URL: URL of the web page
        style: style of the web page
        max_char_length: denotes how many leading characters of the text to be processed by GPT-3
                         Default value of 1000 for development purposes, since GPT-3 is expensive
    The function defines a failed output by default, and updates it if the processing is successful.
    """
    
    output = {
        'URL': request_dict['URL'] if 'URL' in request_dict else '', 
        'style': request_dict['style'] if 'style' in request_dict else '', 
        'output': '',
        'status': 'FAILED'}

    # get the URL
    url = request_dict['URL'] if 'URL' in request_dict else ''
    
    # check validity of url
    if url == '':
        output['output'] = 'ERROR: no URL provided.'
        return output

    # get the style
    style = request_dict['style'] if 'style' in request_dict else ''
    
    if 'is_marked_text' in request_dict and request_dict['is_marked_text']:
        # we use the marked text by the user, instead of scraping the URL
        if 'marked_text' not in request_dict:
            output['output'] = 'ERROR: marked text not provided.'
            return output
        request_dict['marked_text'] = nlp_utils.clean_marked_text(request_dict['marked_text'])
        if len(request_dict['marked_text']) < MIN_MARKED_TEXT_LENGTH:
            output['output'] = 'ERROR: marked text too short.'
            return output
        coarse_text = request_dict['marked_text']
        original_url_webpage = request_dict['marked_text']
    elif 'is_marked_text' not in request_dict or not request_dict['is_marked_text']:    
        # get the text from the URL - or - if supplied, from the HTML.
        # original_url_webpage is simply the downloaded webpage
        # coarse_text is the text to be processed by GPT-3, after it was processed by a cleaning backend
        # such as jusText or Trafilatura
        if request_dict['web_html'] == '':
            coarse_text, original_url_webpage = scraping_utils.url_to_text(url, data_path)
        else:
            coarse_text, original_url_webpage = scraping_utils.html_to_text(request_dict['web_html'])
        if coarse_text == '':
            output['output'] = 'ERROR: time-out or problem cleaning the webpage. Try marking the text you\'re interested in and click the Brush button to Process that text in particular.'
            return output

    # We previously limited the use to English only. For not we allow all languages.
    if False: # nlp_utils.text_not_in_english(coarse_text):
        output['output'] = 'ERROR: We currently only support English.'
        return output

    # Get the structured data from GPT-3
    try:
        response, gpt_credits = get_gpt_summary(coarse_text, style=style, max_char_length=max_char_length, model=model)
    except Exception as e:
        if 'Hijacked' in str(e):
            print('Hijacked error occured. Trying again with variant if exists.')
            try:
                response, gpt_credits = get_gpt_summary(coarse_text, style=style + '_variant', max_char_length=max_char_length, model=model)
            except Exception as e:
                print('Some error occured on second try. Error: ', e)
                response, gpt_credits = '', 0
        else:
            response, gpt_credits = '', 0
            
    if response == '':
        output['output'] = 'ERROR: problem occured. Try changing the style or shorten the text.'
        return output
    elif response.startswith('ERROR'):
        output['output'] = response
        return output

    # convert the structured data to a dictionary
    output["model_output"] = response
    output["output"] = parse_gpt_response(output["model_output"], style=style)

    output["title"] = get_title_for_entry(coarse_text)

    output["cleaned_text"] = coarse_text
    output["original_web"] = original_url_webpage
    output["marked_text"] = request_dict["marked_text"] if 'marked_text' in request_dict else ''

    output["status"] = "SUCCESS"
    output["gpt_credits"] = gpt_credits

    return output

def promptize_qa_list(qa_list, max_prev_questions=2):
    """Promptize the list of questions and answers.
    qa_list: list of tuples (question, answer)
    max_prev_questions: maximum number of previous questions to include in the prompt
    """
    prompt = ''
    for i in range(min(max_prev_questions, len(qa_list))):
        question, answer = qa_list[-1-i][:2]
        prompt = f'Question: {question}\nAnswer: {answer}\n{prompt}'
    return prompt

def get_gpt_answer_to_question(question: str, snippets: list[str], qa_list, text, model=qa_model) -> str:
    """Get response from GPT-3 API.
    question: to be answered using the snippets
    snippets: list of strings that likely contain the answer to the question
    
    The question is put together with the snippets and a prompt, and is sent to GPT3.
    Note: may consider a cheaper model (next cheaper OpenAI: text-curie-001. Can also consider open-source model)
    """
    '''
    After launching, we see that the use case is a little different than what we had in mind.
    Users like to use the chat as ChatGPT rather than asking questions about the text.
    We therefore implement the following change: when the text is sufficiently short, we feed it directly to the model,
    without selecting text based on embeddings.
    '''
    language = nlp_utils.detect_language(text) # either 'en' or not for now (3/4/2023)
    if (len(text) < 9500 and language == 'en') or (len(text) < 5600):
        print('Asking question directly to model, as text is short.')
        response_text = chat_question(question, qa_list, context_text=text, model=model)
        return response_text, ''
    
    #prompt_title = '''You are trying to help a user get an answer to a question based on a document. You are given the question, the first 1000 characters of the text for context and several possibly relevant snippets of text that may contain (or may not) the answer. If the snippets do not contain the answer but you know the answer regardless of them - give the answer, but admit that it is not based on the document (adding \"(not based on the document)\"). If you're not sure about the answer, refuse to give an answer and admit that you're not sure, but again - if you know the answer from elsewhere - say it. Be concise, informative and give only the answer to the question.'''
    prompt_title = '''You are trying to help a user get an answer to a question based on a document. You are given the question, the first 1000 characters of the text for context and several possibly relevant snippets of text that may contain (or may not) the answer. If you are not sure what is the answer, say you're not sure. Be concise, informative and give only the answer to the question.'''
    prompt_title_w_prev_qa = '''You are trying to help a user get an answer to a question. You are given previous answered questions, the new question and several sentences or snippets of text that may contain (or may not) the answer. Try to give the answer to the question. If you are not absolutely sure, say you're not sure. Be concise.'''

    previous_questions_answers_prompt = promptize_qa_list(qa_list)

    question = question.strip()
    question = question[0].upper() + question[1:] # capitalize the first letter of the question
    if question[-1] != '?':
        question += '?'
    output_prompt = 'Answer (either based on the snippets or not):'

    successful_response = False
    number_of_snippets = len(snippets)
    print('Number of snippets: ', number_of_snippets)
    while not successful_response and number_of_snippets > 0:

        text_to_decode = [snip + '\n' for snip in snippets[:number_of_snippets]]
        query_to_model = prompt_title + "\n" + question + '\n' + 'Context:\n' + text[0:1000] + '\nSnippets:\n' + ''.join(text_to_decode) + output_prompt + "\n"
        # print(query_to_model[:100])
        print('query to model ###########################')
        print(query_to_model)
        try:
            response = gpt_completion(query_to_model, max_tokens=512, model=model)
            successful_response = True
        except Exception as e:
            print(e)
            print('Decreasing amount of candidate snippets.')
            number_of_snippets -= 1
        
    if number_of_snippets == 0:
        return 'ERROR: Candidate answer snippets are too long.', ''

    response_text = gpt_response_to_clean_text(response, model)

    return response_text, query_to_model

def qa_about_text(question: str, text: str, url: str, qa_list, top=6, sigma=1, top_answers=4, compact_sentences=5):
    """Get answer to a question about a text.
        question: to be answered using the text
        text: text to be used for answering the question
        url: url of the text (for embedding caching purposes)
        top: number of top similar sentences to use for generating the answer
        sigma: number of sentences around the top similar sentences to use for generating the answer
        top_answers: number of top answers to return
    """
    try:
        cosine_similarities, sentences, embeddings_a, embeddings_q = nlp_utils.get_embeddings(question, text, url, backend="openai", compact_sentences=compact_sentences) # nlp_utils.get_embeddings_qa(question, text)
        print('Got embeddings.')
    except Exception as e:
        print(e)
        return 'ERROR: problem occured. Try again or try selecting another text.', None, None
    
    top = min(top, len(sentences))

    '''
    After getting question-sentence similarities there are a few options
    1) pick top similar sentences
    2) pick top similar sentences and a few sentences around them (sigma)
    3) something else (?)

    We pick first option for now.
    Then, we ask GPT3 to generate an answer to the question based on the similar sentences
    '''
    # get top_answers sentences whose cosine_similarities is largest but their length is larger than 10 characters
    for i in range(len(cosine_similarities)):
        if len(sentences[i]) < 10:
            cosine_similarities[i] = 0
    top_sentences_locations = np.argsort(cosine_similarities)[-top:]

    sentences_islands = nlp_utils.find_islands(top_sentences_locations, sigma=sigma, length=len(sentences))
    top_sentences = [str(j+1) + '. ' + ' '.join([sentences[i] for i in island]) for j, island in enumerate(sentences_islands)]
    top_sentences = [sent.replace('\n', ' ') for sent in top_sentences]
    top_sentences = [re.sub(r'\s{2,}', ' ', sent) for sent in top_sentences]

    response_text, query_to_model = get_gpt_answer_to_question(question, top_sentences, qa_list, text)
    response_text = response_text.strip() # basic cleaning

    supporting_sentences = nlp_utils.get_supporting_sentences(sentences_islands, embeddings_a, response_text, sentences, top_answers)
    supporting_quote = '...' + '... '.join(supporting_sentences) + '...'

    # replace \n in supporting_quote with space
    supporting_quote = supporting_quote.replace('\n', ' ')

    # replace multiple spaces with one space
    supporting_quote = re.sub(r'\s{2,}', ' ', supporting_quote)

    return response_text, query_to_model, supporting_quote

def prepare_qa_for_chat(question, answer):
    if question.startswith('/chat '):
        question = question[5:]
    else:
        question = question + '\n(Based on an attached document - redacted)'
    return question, answer

def chat_question(question, qa_list, context_text='', model="gpt-3.5-turbo"):
    prev_msgs = []
    for qa in qa_list:
        prev_question, prev_answer = qa[:2]
        prev_question, prev_answer = prepare_qa_for_chat(prev_question, prev_answer)
        prev_msgs.append({"role": "user", "content": prev_question})
        prev_msgs.append({"role": "assistant", "content": prev_answer})
    if context_text == '':
        query = question
    else:
        query = question + '\nContext text:\n' + context_text
    
    prev_msgs.append({"role": "user", "content": query})
    
    try:
        response = gpt_completion('', max_tokens=768, model='gpt-3.5-turbo', prev_msgs=prev_msgs)
    except Exception as e:
        print(e)
        return 'ERROR: problem occured. Try again or try selecting another text.'
    if 'choices' not in response:
        return 'ERROR: problem occured. Try again or try selecting another text.'
    
    answer = response['choices'][0]['message']['content']
    answer = answer.strip()

    return answer