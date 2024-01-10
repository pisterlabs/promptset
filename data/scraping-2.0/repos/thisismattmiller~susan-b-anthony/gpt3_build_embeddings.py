import openai
import pickle
import glob
import json
import util
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))


EMBEDDING_MODEL = "text-embedding-ada-002"



total_daybook = len(list(glob.glob('daybook-and-diaries-1856-1906-daybook-1*/*.json')))
done_counter = 0
for file in glob.glob('daybook-and-diaries-1856-1906-daybook-1*/*.json'):
    done_counter+=1
    print(done_counter, '/',total_daybook)
    dir = file.split('/')[-2]
    file_id = int(file.split('/')[-1].replace('.json',''))

    data = json.load(open(file))
    # for the daybooks we use the extracted text which doesn't have the date
    if 'gpt' in data:
        if 'gpt3.5-daybook-json' in data['gpt']:

            for entry in data['gpt']['gpt3.5-daybook-json']:
                if 'embedding' in entry:
                    print('skip entry')
                    continue

                if entry['fullText'] == None:
                    continue

                if count_tokens(entry['fullText']) > 40:
                    print(file)
                    
                    text = entry['fullText'].replace('\n',' ')
                    text = util.clean_up_transcribed_text(text)

                    print(text)
                    result = openai.Embedding.create(
                      model=EMBEDDING_MODEL,
                      input=text
                    )
                    entry['embedding'] = result["data"][0]["embedding"]

            json.dump(data,open(file,'w'),indent=2)
            




total_writtings = len(list(glob.glob('anthony-speeches-and-other-writings-resources/*.json')))
done_counter = 0

for file in glob.glob('anthony-speeches-and-other-writings-resources/*.json'):

    done_counter+=1
    print(done_counter, '/',total_writtings)
    print(file)
    file_id = file.split('/')[-1].replace('.json','')

    data = json.load(open(file))
    # for the daybooks we use the extracted text which doesn't have the date
    for block in data:
        if len(block['text']) > 0:

            text = block['text']
            text = text.replace('\n',' ')
            text = util.clean_up_transcribed_text(text)

            result = openai.Embedding.create(
              model=EMBEDDING_MODEL,
              input=text
            )
            block['embedding'] = result["data"][0]["embedding"]

    json.dump(data,open(file,'w'),indent=2)
    





total_writtings = len(list(glob.glob('anthony-correspondence-resources/*.json')))
done_counter = 0

for file in glob.glob('anthony-correspondence-resources/*.json'):

    done_counter+=1
    print(done_counter, '/',total_writtings)
    print(file)
    file_id = file.split('/')[-1].replace('.json','')

    data = json.load(open(file))
    # for the daybooks we use the extracted text which doesn't have the date
    all_text = ""
    for item in data['items']:
        if 'full_text' in item:

            text = item['full_text']
            text = text.replace('\n',' ')
            text = util.clean_up_transcribed_text(text)
            all_text=all_text + ' ' + text


    if count_tokens(all_text) <= 1000:
        print("<1000")
    else:
        print("bigger",count_tokens(all_text))
    

    result = openai.Embedding.create(
      model=EMBEDDING_MODEL,
      input=all_text
    )
    data['embedding'] = result["data"][0]["embedding"]

    json.dump(data,open(file,'w'),indent=2)



