import os
import json
import argparse
import transformers
from tqdm import tqdm
import multiprocessing
from langchain import RecursiveCharacterTextSplitter

# Function to chunk an article
def chunk_text_langchain(article, tokenizer):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 256,
        chunk_overlap = 20,
        length_function = lambda str: len(tokenizer.encode(str))
    )

    return (
        [doc.page_content for doc in splitter.create_documents([article['text']])], 
        article['id']
    )

def main(args):
    # Validate the existence of the input directory
    if not os.path.isdir(args.input):
        raise ValueError(f'{args.input} is not a valid path')
    print(f'Processing {args.input}')

    # Check for the existence of a metadata file already
    metadata_filepath = os.path.join(args.input, f'meta_{args.metadata_suffix}.json')
    if os.path.exists(metadata_filepath):
        print('Metadata already extracted for the current directory')
        return
    
    # Parse articles from files
    articles = {}
    for filename in os.listdir(args.input):
        # Skip metadata files
        if 'wiki_' not in filename: continue

        # Process the wiki file
        with open(os.path.join(args.input, filename), 'r') as fin:
            for line in fin:
                if (article := json.loads(line))['text'] != '':
                    if article['id'] not in articles:
                        articles[article['id']] = article
                    else:
                        raise ValueError(f"Not-unique ID {article['id']}")

    # Tokenizer and model instantiation
    tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/msmarco-distilbert-base-v4')

    # Suppress tokenization length warnings
    transformers.utils.logging.set_verbosity_error()

    # Chunk the articles
    progress_bar = tqdm(total=len(articles))
    with multiprocessing.Pool(processes=32) as pool:
        async_results = [
            pool.apply_async(
                chunk_text_langchain, 
                args=(articles[article_id], tokenizer), 
                callback=lambda _: progress_bar.update()
            ) for article_id in articles
        ]

        # Collect async results
        for async_res in async_results:
            chunks, article_id = async_res.get()
            articles[article_id]['chunks'] = chunks

        # Close out worker pool
        pool.close()
        pool.join()
    
    # Close out progress bar
    progress_bar.close()

    # Unsuppress warnings
    transformers.utils.logging.set_verbosity_warning()

    # Write out the article metadata to json file
    metadata = {}
    for article_id in articles:
        articles[article_id]['chunks_count'] = len(articles[article_id]['chunks'])
        metadata[article_id] = {key: articles[article_id][key] for key in articles[article_id] if key not in {'text', 'chunks'}}
    with open(metadata_filepath, 'w') as fout:
        json.dump(metadata, fout)

if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description = 'Chunks and embeds the articles in the input directory.'
    )
    parser.add_argument(
        '--input', 
        type = str,
        required = True,
        help = 'Path to directory with article files.'
    )
    parser.add_argument(
        '--metadata_suffix', 
        type = str,
        required = True,
        help = 'Suffix to append to generated metadata filenames.'
    )

    # Parse arguments
    args = parser.parse_args()

    # Launch script
    main(args)
