import pysrt
import openai
import time
import json
import os
def process_text(text, target_language, api_key, 
                   playrole,task,max_tokens=250, max_retries=3, retry_interval=3):
    openai.api_key = api_key
    retries = 0
    
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": f"{playrole}"
                    },
                    {
                
                        "role": "user",
                        "content": f'{task} {target_language}: "{text}"'
                    }
                ]
            )
            translated_text = response['choices'][0]['message']['content']
            return translated_text.strip()
        except Exception as e:
            print(f"Error occurred while translating (attempt {retries + 1}): {e}")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_interval)
            else:
                print(f"Translation failed after {max_retries} attempts. Waiting for breakpoint reconnect.")
                return None
def load_progress(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        with open(file_path, 'w') as f:
            json.dump({}, f)
        return {}

def save_progress(file_path, progress):
    with open(file_path, 'w') as f:
        json.dump(progress, f)

def load_glossary(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def find_glossary_terms(text, glossary):
    terms_found = []
    for term, definition in glossary.items():
        if term in text:
            terms_found.append((term, definition))
    return terms_found

def SubsTranslator(src, des, targetlanguage, API, playrole='',task='Translate the following text to', relative_path='', progress_file='', glossary={}):
    if os.path.exists(des):
        subs = pysrt.open(des)
    else:
        subs = pysrt.open(src)
    progress = load_progress(progress_file)
    start_index = progress.get(relative_path, 0)
    for idx, sub in enumerate(subs[start_index:]):
        original_text = sub.text
        glossary_terms = find_glossary_terms(original_text, glossary)
        if glossary_terms:
            glossary_text = " ".join([f"{term}: {definition}" for term, definition in glossary_terms])
            playrole_with_glossary = f"{playrole} ,The following is a glossary: {glossary_text}"
            print(f"Found glossary terms: {glossary_terms}")
        else:
            playrole_with_glossary = playrole
        translated_text = process_text(original_text, targetlanguage, API, playrole_with_glossary,task).replace('。', ' ').replace('，', ' ').replace('\"', '').replace('“', '').replace('”', '')
        if translated_text:
            sub.text = f"{translated_text}\\n{original_text}"
            subs.save(des, encoding='utf-8')
            print(f"Translated subtitle index {start_index + idx}: {original_text} -> {translated_text}")
            progress[relative_path] = start_index + idx + 1
            save_progress(progress_file, progress)
        else:
            print(f"Failed to translate subtitle index {start_index + idx}: {original_text}")
        time.sleep(0.1)

def SubsSummarizer(src, des, target_language, api_key, playrole='', task='Summarize the following text to', start_index=0, end_index=None):
    subs = pysrt.open(src)
    if end_index is None:
        end_index = len(subs)
    long_text = " ".join([sub.text for sub in subs[start_index:end_index]])
    summarized_text = process_text(long_text, target_language, api_key, playrole, task)
    if summarized_text is not None:
        with open(des, 'w') as outfile:
            outfile.write(summarized_text)
        print(f"Summarized {summarized_text} to {des}.")
    else:
        print("Summarization failed.")
