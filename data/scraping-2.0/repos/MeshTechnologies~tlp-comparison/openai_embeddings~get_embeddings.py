import os
import json
import time
import openai
from tqdm import tqdm


def get_embeddings(data, engine, break_lines):
    if break_lines:
        for i in range(len(data)):
            data[i] = data[i].replace('\n', ' ')

    response = openai.Embedding.create(input=data, engine=engine)['data']
    return [line['embedding'] for line in response]


def build_embedding_dataset(path, engine, batch, break_lines):
    if not os.path.exists(os.path.join('openai_embeddings', 'output', engine)):
        os.mkdir(os.path.join('openai_embeddings', 'output', engine))

    for subset in os.listdir(path):
        print(f'Generating embeddings for {subset}...')
        languages = os.listdir(os.path.join(path, subset))
        with tqdm(total=len(languages) * len(os.listdir(os.path.join(path, subset, languages[0])))) as pbar:
            for language in languages:
                if not language.startswith('.'):
                    files = os.listdir(os.path.join(path, subset, language))
                    for i in range(0, len(files), batch):
                        data_batch = []

                        for filename in files[i:i + batch]:
                            pbar.update(1)
                            if not filename.startswith('.'):
                                data_batch .append(open(os.path.join(path, subset, language, filename), 'r').read())

                        embeddings = get_embeddings(data_batch, engine, break_lines)

                        # rate limit
                        time.sleep(1)

                        with open(os.path.join('openai_embeddings', 'output', engine, f'{subset}.jsonl'), 'w+') as f:
                            for line in embeddings:
                                f.write(json.dumps({'embeddings': line, 'label': language}, ensure_ascii=False) + '\n')


def main():
    # load user key for openAI
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # create directory for output
    if not os.path.exists(os.path.join('openai_embeddings', 'output')):
        os.mkdir(os.path.join('openai_embeddings', 'output'))

    # programming language classification dataset
    build_embedding_dataset(
        path='56_lang_sampled_dataset_weak_cobol',
        engine='code-search-ada-code-001',
        batch=512,
        break_lines=False
    )


if __name__ == '__main__':
    main()
