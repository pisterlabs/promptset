'''
This version applys ocr on symbol bbox w/ buffer
match the ocr words from buffered bbox and description

'''
import openai
from openai import OpenAI
import base64
import requests
import json
import collections
import os
import sys
from symbol_bbox_ocr import get_symbol_names
# get_symbol_names_within_roi_with_buffer_v2, get_symbol_names_within_roi_with_buffer

def remove_unicode(text):
    # Create an empty string to store the cleaned text
    cleaned_text = ''
    
    for char in text:
        if char < 128:  # Check if the character is ASCII
            cleaned_text += chr(char)
        elif len(cleaned_text) > 0 and cleaned_text[-1] != ' ':
            cleaned_text += ' '
    return cleaned_text

def str2json(string):
    json_str = ''
    s_ind = 0
    e_ind = len(string)
    for i, ch in enumerate(string):
        if ch == '{':
            s_ind = i
            break
    for i, ch in enumerate(string[::-1]):
        if ch == '}':
            e_ind = len(string) - i
            break
#     json_str = string[s_ind:e_ind].encode("ascii", "ignore").decode()
    json_str = string[s_ind:e_ind]
    dic = {}
    for item in json_str.split('},'):
        if "}" != item.strip()[-1]:
            dict_str = item.strip()+'}'
        else:
            dict_str = item.strip()
        try:
            temp_dict = eval(dict_str)
        except SyntaxError as e:
            print('GPT4 result is not a json')
            return SyntaxError('GPT4 result is not a json')
        try:
            dic[temp_dict['symbol name']] = temp_dict['description']
        except KeyError as e:
            print('symbol name/description is not the key')
            return KeyError('symbol name/description is not the key')
    return dic

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def call_gpt_api(image_path, max_attempts=10, delay=5):
    client = OpenAI()
    base64_image = encode_image(image_path)
    
    for attempt in range(max_attempts):
        print(f'attempt {attempt}')
        try:   
            if attempt >= 8:
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                    {
                      "role": "user",
                      "content": [
                        {"type": "text", "text": "Please extract information in the image? Please display the response as a list of dictionaries, the dictionary has two keys: symbol name and description. Please make sure the description is completed. The results should only have the dictionaries."},
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                          },
                        },
                      ],
                    }
                  ],
                  max_tokens=4096,
                )

            else:
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                    {
                      "role": "user",
                      "content": [
                        {"type": "text", "text": "Please extract all symbols and corresponding descriptions in the image? Please display the response as a list of dictionaries, the dictionary has two keys: symbol name and description. The \"symbol name\" key is for symbols. The \"description\" key is for the description corresponding to the symbol. Please make sure the description is completed. The results should only have the dictionaries."},
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                          },
                        },
                      ],
                    }
                  ],
                  max_tokens=4096,
                )
            
            extract_res = response.choices[0].message.content   
            gpt_json = str2json(extract_res)
            if not isinstance(gpt_json, Exception):
                return gpt_json
        except (openai.InternalServerError, openai.OpenAIError, ValueError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt + 1 < max_attempts:
                time.sleep(delay)
            else:
                raise
    return ValueError('GPT4 result is not a json')


def match_str(query_words, source_str):
    num_matched = 0
    for word in query_words:
        if word in source_str:
            num_matched += 1.0
    return num_matched/len(query_words)

def match_symbol(query_str, gpt_dict, matched_flag):
    matched_score = 0
    
    query_words = query_str.split(' ')
    
    matched_symbol_name = None
    
    for sym, des in gpt_dict.items():
        if len(des) > 500:
            score1 = match_str(query_words, des[:len(des)//2])
        else:
            score1 = match_str(query_words, des[:])
        score2 = match_str(query_words, sym)
        score = score1 + score2
        if score > matched_score: # and not matched_flag[sym]:
            matched_score = score
            matched_symbol_name = sym
        
    return matched_symbol_name

def main(image_name, map_dir, all_sym_bbox, image_dir='/data/weiweidu/gpt4/map_legend_gpt_input_hw', output_dir='./res'):
    ''' 
    '''
    # Path to gpt input image (cropped image)
    image_path = f'{image_dir}/{image_name}.png'
    map_name = '_'.join(image_name.split('_')[:-4])

    # ========== check the existence of map image, json ===================
    if not os.path.isfile(image_path):
        print(f'*** {image_name}.tif does not exist! ***')
        sys.exit()
    
    # ========== get texts in the buffered symbol bbox ===================
    symbol_texts, symbol_bboxes = get_symbol_names(image_dir, image_name, map_dir, all_sym_bbox)
    print(symbol_bboxes)
    
    # ========== get json from gpt4 API {'symbol_name': 'description'} ===================
    json_res = call_gpt_api(image_path)
    
    if isinstance(json_res, Exception):
        sys.exit()
    
    # ========== match res from gpt4 API and symbol bbox ===================
    sym_des_dict = {}
    matched_dict = {}
    for sym in json_res.keys():
        matched_dict[sym] = False

    for i, symbol in enumerate(symbol_texts):
        matched_symbol = match_symbol(symbol, json_res, matched_dict)
        matched_dict[matched_symbol] = True
        if matched_symbol is not None:
            sym_des_dict[str(symbol_bboxes[i])] = {'description': json_res[matched_symbol], 'symbol name': matched_symbol}


    with open(f"{output_dir}/{image_name}.json", "w") as outfile:
        json.dump(sym_des_dict, outfile)
    print(f'*** saved result in {output_dir}/{image_name}.json ***')
    return


