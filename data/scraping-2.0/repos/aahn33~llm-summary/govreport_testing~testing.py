import json
import sys
sys.path.append('../')

from pathlib import Path
from langchain.chat_models import ChatOpenAI
from rouge import Rouge

from map_and_refine.map_reduce import MapTextSummarizer
from relevancy_score_tagging.relevancy import RelevancyTagger


def calculate_rouge(title, summary, reference):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)[0]
    score_file = open("%s.txt"%title, "w")
    score_file.write(f"ROUGE-1: Precision = {scores['rouge-1']['p']}, Recall = {scores['rouge-1']['r']}, F1 = {scores['rouge-1']['f']}\n")
    score_file.write(f"ROUGE-2: Precision = {scores['rouge-2']['p']}, Recall = {scores['rouge-2']['r']}, F1 = {scores['rouge-2']['f']}\n")
    score_file.write(f"ROUGE-L: Precision = {scores['rouge-l']['p']}, Recall = {scores['rouge-l']['r']}, F1 = {scores['rouge-l']['f']}\n")
    score_file.close()

# Sample test using a single extracted file
with open(Path('extracted/98-696.json'), encoding='utf-8') as f:
    # Retrieve testing data
    data = json.loads(f.read())

    title = data['title']
    ground_truth = data['summary']
    full_text = data['full_text']

    # Initialize LLM
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0, openai_api_key="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",
                    model_name=model_name)
    
    # Filter for relevant chunks in text
    tagger = RelevancyTagger(llm, model_name, 0.8, 500)
    relevant_text = tagger.tag(full_text, title)

    # Get output from full text map_reduce
    summarizer = MapTextSummarizer(llm=llm, model_name=model_name)
    full_summary, total_tokens_used = summarizer.process(full_text)

    print(full_summary)
    print(f"Total tokens used: {total_tokens_used}")

    # Get output from relevant text map_reduce
    relevant_summary, total_tokens_used = summarizer.process(relevant_text)
    print(relevant_summary)

    # Perform rouge test
    calculate_rouge('test_full', full_summary, ground_truth)
    calculate_rouge('test_relevant', relevant_summary, ground_truth)




