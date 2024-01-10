import numpy as np
import pandas as pd
import openai
import argparse
import random
import time
import os
from synthwiki_utils import checkCorrectness, insertIntoJunk, genJunkContext, hash_string, check_file_exists
import transformers

##  python eval_gpt3.py --answer_type docs --include_true_doc 0 --result_dir gpt_3/madlibs1_1000_doc/ --junk_size 1000

## python eval_gpt3.py --result_dir gpt_3/madlibs1_no_junk/ --junk_size 1000 --no_junk 1
parser = argparse.ArgumentParser(description="A simple argparse example.")
parser.add_argument('--input_file', 
                    default='/home/ubuntu/cut_attention/data/madlibs/madlibs1.csv',
                    help='Where questions?')
parser.add_argument('--result_dir', 
                    help='Where results to save?', required=True)
parser.add_argument("--openai_key", help="UR KEY BOR")
parser.add_argument("--junk_size", default=16000, type=int, help="How much junk")
parser.add_argument('--no_junk', default=0, type=int, help="Clean run with just the single doc")
parser.add_argument("--include_true_doc", default=1, help="Should we include true doc?!")
parser.add_argument("--answer_type", default='qa', help="Ask question or ask about doc?")
parser.add_argument('--insert_place', 
                    default='random',
                    help='Should I put real doc at (max_pos / 2) or random?')
parser.add_argument("--sample_n", default=100, type=int)
parser.add_argument("--model", default='gpt-3.5-turbo-16k', type=str)

args = parser.parse_args()

print("Loading Llama tokenizer to do fair comparisons")
tokenizer = transformers.AutoTokenizer.from_pretrained('togethercomputer/LLaMA-2-7B-32K')
print("Loaded tokenizer.")

assert args.answer_type in ['qa', 'docs'], "Answer type needs to be qa or docs"
if args.answer_type == 'qa':
    assert args.include_true_doc == 1, "You have qa on but aren't including the real document, rejected."

os.makedirs(args.result_dir, exist_ok=True)

openai.api_key = args.openai_key

raw = pd.read_csv(args.input_file)
all_contexts = np.unique(raw['context'].values)

## ADD A SAMPEL
re_ordered = raw.sample(args.sample_n)
real_context = re_ordered['context'].values
real_question = re_ordered['question'].values
real_answer = re_ordered['answer'].values

example_junk = genJunkContext(real_context.tolist(), limit=2500, tokenizer=tokenizer)
print(example_junk)

def askGPTWhichDocs(question, supporting_docs, model):
    doc_string = ""
    
    doc_ids = []

    for d in supporting_docs:
        doc_id = hash_string(d, len=8)
        doc_ids.append(doc_id)
        doc_string += f"Document [{doc_id}]: {d} \n \n"

    prompt = f"""I will show you a list of documents and a question.
Each document has a document id. Your goal will be to identify which documents are 
relevant to answering the question. Some of them will not be.

Here is an example of how a document will appear:

EXAMPLE DOCUMENT
DOCUMENT [document_id]: "Document Text"

DOCUMENTS
{doc_string}

Here is a question: {question}

Please output a comma seperated list of document ids that contain the answer to the question.
If no documents in the list contain the answer, just say "none are relevant." """

    sys_prompt = """Your job is to help people. Please do that."""

    retry_limit = 10
    retry_count = 0
    while retry_count < retry_limit:
        try:
            response = openai.ChatCompletion.create(
                messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': prompt},
            ],
            model=model
            )
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            retry_count += 1
            time.sleep(60)

    if retry_count == retry_limit:
        answer = "Reached maximum retry limit."
        return 'something failed'
    
    answer = response['choices'][0]['message']['content']
    
    docs_relevant = []
    for (id, doc) in zip(doc_ids, supporting_docs):
        if id in answer:
            docs_relevant.append(doc)

    return answer, docs_relevant

def askGPT(question, supporting_docs, model):
    doc_string = ""
    
    doc_ids = []

    for d in supporting_docs:
        doc_id = hash_string(d, len=8)
        doc_ids.append(doc_id)
        doc_string += f"DOCUMENT: {d} \n\n"

    prompt = f"""Here are some documents you will use to answer a question. Some of the documents may not be relevant.\n\nDOCUMENTS{doc_string}Here is a question:\n\nQUESTION\n{question}\n\nPlease return only the answer to the question. Answer concisely.
    """

    sys_prompt = """You are a helpful AI assistant."""

    retry_limit = 10
    retry_count = 0
    while retry_count < retry_limit:
        try:
            response = openai.ChatCompletion.create(
                messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': prompt},
            ],
            model=model
            )
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            retry_count += 1
            time.sleep(60)

    if retry_count == retry_limit:
        answer = "Reached maximum retry limit."
        return 'something failed'
    
    answer = response['choices'][0]['message']['content']

    return answer


if __name__ == "__main__":
    ## Not parallelizing this since you hit GPT rate limit basically instantly, one at a time is fine
    for (question, context, answer) in zip(real_question, real_context, real_answer):
        qhash = hash_string(question)
        fname = f'{args.result_dir}/{qhash}_gpt3_{args.answer_type}.csv'
        
        if check_file_exists(fname):
            print("lol this exists")
        else:
            junk_contexts = [c for c in all_contexts if c != context]

            context_to_use = genJunkContext(
                junk_contexts, 
                limit=args.junk_size, 
                tokenizer=tokenizer,
            )
            if args.no_junk:
                supp_docs = [context]
                pos_to_insert = 0
            
            random.shuffle(context_to_use)
            if args.include_true_doc == 1:
                supp_docs, pos_to_insert = insertIntoJunk(context_to_use, context, args.insert_place)
            else:
                supp_docs = context_to_use
                pos_to_insert = 0
                
            if args.answer_type == 'docs':
                model_answer, docs_relevant = askGPTWhichDocs(question, supp_docs, model=args.model)
                print(f"Question: {question} | Answer: {answer}")
                print(f"Model answer: {model_answer}")
                print(f"{docs_relevant}")
                row = pd.DataFrame({
                    'question': question,
                    'junk_size': args.junk_size,
                    'n_relevant_documents': len(docs_relevant),
                    'model_answer': model_answer,
                    'include_true_doc': args.include_true_doc
                }, index=[0])
            
            if args.answer_type == 'qa':
                model_answer = askGPT(question, context_to_use, model=args.model)

                print(f"Question: {question} | Answer: {answer}")
                print(f"Model answer: {model_answer}")

                correct = checkCorrectness(model_answer, answer)
                row = pd.DataFrame({
                    'question': question,
                    'junk_size': args.junk_size,
                    'correct': correct,
                    'model_answer': model_answer,
                    'include_true_doc': args.include_true_doc,
                    'doc_position': pos_to_insert,
                    'model': args.model
                }, index=[0])

            row.to_csv(fname, index=False)
