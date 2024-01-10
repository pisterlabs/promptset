from datasets import load_dataset
from datetime import datetime
import tensorflow_datasets as tfds
import re
import time
import openai


def get_quac(split='train'):
    quac = load_dataset('quac')[split]
    return quac

def get_wiki(dataset_name='wikipedia/20230601.en'):
    builder = tfds.builder(dataset_name, file_format='tfrecord')
    builder.download_and_prepare()
    dataset = builder.as_dataset()
    wiki = dict()
    for sample in tfds.as_numpy(dataset['train']):
        wiki[sample['title'].decode('utf-8')] = sample['text'].decode('utf-8')
    return wiki

def get_longcacti_quac(openai_api_key):
    openai.api_key = openai_api_key
    quac = get_quac()
    wiki = get_wiki()
    def get_sample_wiki_fulltext_refs(sample):
        title = sample['wikipedia_page_title']
        wiki_page_info = dict()
        fulltext = wiki.get(title, None)
        if fulltext is not None:
            refs_title = [title for line in re.split('\nReferences|\nSee also', fulltext)[1:] for title in line.split('\n') if title.strip() != '']
            wiki_page_info = {'wikipedia_page_text':fulltext, 'wikipedia_page_refs':[{'title': title, 'text': wiki[title]} for title in refs_title if wiki.get(title, None) is not None]}
        return wiki_page_info

    def get_sample_gpt4_answer(sample, sleep=60):
        gpt4_answers_info = dict()
        try:
            context = sample['context']
            questions = sample['questions']
            answers = sample['answers']['texts']
            gpt4_answers = []
            gpt4_answers_check = []
            prompt = f"Context:\n{context}\nAnswer the questions based only on the information from the context above.\nQuestion: {questions[0]}"
            for i in range(len(questions)):
                messages = [{"role": "user", "content": f"{prompt}"}]
                for j in range(len(gpt4_answers)):
                    messages.append({"role": "assistant", "content": gpt4_answers[j]})
                    messages.append({"role": "user", "content": questions[j + 1]})
                response = openai.ChatCompletion.create(model="gpt-4", messages=messages,)
                gpt4_answers.append(response["choices"][0]["message"]["content"])
                provided_answer = ["The context does not provide any information about the question. " if answer[0] == 'CANNOTANSWER' else answer[0] for answer in answers]
                check_prompt = f"Question: {questions[i]}\nAnswer 1: {gpt4_answers[i]}\nAnswer 2: {provided_answer[i]}\nAre the two answers similar in meaning? Answer yes, no or neutral."
                response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": f"{check_prompt}"}],)
                gpt4_answers_check.append(response["choices"][0]["message"]["content"])
                gpt4_answers_info = {'gpt4_answers': gpt4_answers, 'gpt4_answers_consistent_check': gpt4_answers_check}
        except:
            print(f'{datetime.now().strftime("%H:%M:%S")}, failed for: {sample["dialogue_id"]}, sleeping {sleep}s')
            time.sleep(sleep)
        return gpt4_answers_info

    longcacti_quac = []
    for sample in quac['train']:
        sample.update(get_sample_wiki_fulltext_refs(sample))
        sample.update(get_sample_gpt4_answer(sample))
        longcacti_quac.append(sample)



if __name__ == "__main__":
    get_longcacti_quac(openai_api_key='xxx')

