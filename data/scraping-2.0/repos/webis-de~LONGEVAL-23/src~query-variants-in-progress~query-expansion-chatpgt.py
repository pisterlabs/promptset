#!/usr/bin/python3
import json
import ir_datasets

target_file = 'query-expansions-from-chatgpt-raw.json'
prompts = json.load(open('query-expansion-prompts.json'))
datasets = ['longeval/train', 'longeval/heldout', 'longeval/a-short-july', 'longeval/b-long-september']
queries = sorted(list(set([q.text.strip() for d in datasets for q in ir_datasets.load(d).queries_iter()])))


def process_query(query):
    import openai
    print(f'Process Query: {query}')
    
    request_prompt = "2"
    request = prompts[request_prompt].replace('<ORIGINAL_QUERY>', query)
    ret = {'request': request, 'request_prompt': request_prompt}
    ret['gpt-3.5-turbo-response'] = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request}
        ]
    )

    print(f'Response: {ret}')
    
    return ret
    

def main(num=10):
    performed = 0
    ret = json.load(open(target_file))
    
    for query in queries:
        if query in ret.keys():
            continue
        
        try:
            ret[query] = process_query(query)
            performed += 1
        except Exception as e:
            print(e)
            break
        
        if performed > num:
            break

    json.dump(ret, open(target_file, 'w'))


if __name__ == '__main__':
    main(1000)
