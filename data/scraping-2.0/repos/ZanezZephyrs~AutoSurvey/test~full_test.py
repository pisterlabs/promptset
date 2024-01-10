from AutoSurvey.llm_inference.openai_inference import OpenAIInference
from AutoSurvey.searchers.semantic_scholar import SemanticScholarSearcher
from AutoSurvey.filters.field_in_list_filter import FieldInListFilter
from AutoSurvey.filters.field_numeric_filter import FieldNumericFilter
from AutoSurvey.pdf_generation.pdf_generator import PDFGenerator
from AutoSurvey.llm_inference.query_augmentor import QueryAugmentor
from AutoSurvey.ranker import MonoT5
from AutoSurvey.pdf_extraction.pdf_extractor import get_pdf_windows
from tqdm.notebook import tqdm
import argparse
import json
import logging
import os
import tempfile

paper_template="""- Paper {I}
Title:{TITLE}
Content: {CONTENT}
"""

template_messages=[
    {"role": "system", "content": "You Are an expert in scientific literature review, your job is, given a series of papers and theirs summaries, write a paragraph with a given title citing relevant information in the traditional format (e.g [1,2,3] [1]) from the provided papers"},
]

#create debug logger that writes to file
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('debug.log')
fh.setLevel(logging.DEBUG)
debug_logger.addHandler(fh)


searcher=SemanticScholarSearcher()

agent=OpenAIInference(api_key="", engine="gpt-3.5-turbo")

query_augmentor=QueryAugmentor(agent=agent)

def search_one_query(query, filters=None, rerank=False, model=None):
  params={
      "query": query,
      "limit": 10, # max 100
      "fields": "title,authors,citationCount,year,tldr,url,abstract,influentialCitationCount"
  }

  # compose as many filters as you like, they are applied in sequence

  if filters is None:
    filters=[
        # FieldNumericFilter("citationCount", lower_bound=2), # at least 10 citations
        # FieldNumericFilter("authors", lower_bound=2), # at least 2  authors
        FieldNumericFilter("year", lower_bound=2020, upper_bound=2023), # between 2020 and 2023
    ]

  docs = searcher.search(params, filters=filters)
  if len(docs) == 0:
    debug_logger.debug(f"query: {query} returned 0 results with filters {filters}")
    return []
  augmented_docs = []
  for doc in docs:
    if rerank:
      pdf = searcher.download_pdf(doc["paperId"])
      if pdf:
        temp_file = tempfile.NamedTemporaryFile()
        temp_file_path = temp_file.name
        temp_file.write(pdf)
        windows = get_pdf_windows(temp_file_path)
        temp_file.close()
        augmented_docs.extend([{"title": doc["title"], "content": window} for window in windows])
        debug_logger.debug(f"extracted {len(windows)} windows from pdf {doc['title']}\n")
        continue
    if doc["abstract"]:
      augmented_docs.append({"title": doc["title"], "content": doc["abstract"]})
    elif doc["tldr"] and doc["tldr"].get("text"):
      augmented_docs.append({"title": doc["title"], "content": doc["tldr"]["text"]})
    else:
      continue
    scores = model.rescore(query, [doc["content"] for doc in augmented_docs])
    assert len(scores) == len(augmented_docs), "The number of scores should be the same as the number of documents"
    augmented_docs = [x for _, x in sorted(zip(scores, augmented_docs), key=lambda x: x[0], reverse=True)]

  debug_logger.debug(f"query: {query} returned {len(docs)} results with filters {filters}")

  return augmented_docs

def query_to_documents(query, filters=None, augment_query=True, rerank=False, model=None):

  if augment_query:
    queries= query_augmentor.augment_queries(query)
    print("augmentation resulted in the following queries", queries)
    debug_logger.debug(f"augmentation resulted in the following queries {queries}")

    results=[]
    for query in queries:
      results.extend(search_one_query(query, filters=filters, rerank=rerank, model=model))

  else:
    results=search_one_query(query, filters=filters, rerank=rerank, model=model)

  return results
  

def documents_to_section(results, query):

  user_prompt=""
  current_paper_number=0
  for i,result in enumerate(results):
    current_paper_number+=1

    if current_paper_number>=12:
      break

    paper_text=paper_template.format(I=i+1, CONTENT=result["content"], TITLE=result["title"])
    user_prompt+=paper_text + "\n"

  user_prompt+= "Paragraph Subject: " + query

  current_msgs=template_messages.copy()

  current_msgs.append({
      "role": "user",
      "content": user_prompt
  })

  debug_logger.debug(f"prompt for model {user_prompt}")
  response=agent.complete(current_msgs)
  debug_logger.debug(f"response from model {response}", )

  return response


# python -m test.full_test --survey_data C:\Users\Thiago\Documents\projeto_final_ia368\AutoSurvey\data\dataset\survey_1.json --out_path proc5/proc5.json
if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--survey_data", type=str, required=True, help="Path to the json dataset")
  argparser.add_argument("--out_path", type=str, default="proc1.json", help="Path to save the evaluation results")
  argparser.add_argument("--reranker", action="store_true", help="Whether to use the reranker or not")
  argparser.add_argument("--reranker_model", type=str, default="castorini/monot5-large-msmarco", help="Path to save the evaluation results")

  args = argparser.parse_args()

  if not os.path.exists(args.out_path):
    from pathlib import Path
    os.makedirs(Path(args.out_path).parent, exist_ok=True)
    

  with open(args.survey_data, "r") as f:
      data = json.load(f)

  
  survey_title=list(data.keys())[0]
  sections=[key for key in data[survey_title].keys() if data[survey_title][key]["content"] != "" and key not in ["Introduction", "Conclusion"]]

  print(sections)

  sections_data={survey_title: {}}

  debug_logger.debug(f"survey title -> {survey_title}")

  model = None
  if args.reranker:
    model = MonoT5(args.reranker_model, fp16=True)

  for section in tqdm(sections, desc="Processing sections"):
    query=survey_title + " - "+ section
    documents=query_to_documents(query, filters=None, rerank=args.reranker, model=model)
    if len(documents)==0:
      print("skipping section, no relevant documents found for query", query)
      continue
    
    if not args.reranker:
      documents=sorted(documents, key=lambda x: x["influentialCitationCount"], reverse=True)

    
    section_text=documents_to_section(documents, query)
    sections_data[survey_title][section]={
      "content": section_text
    }
  print(sections_data)


  with open(args.out_path, "w") as f:
      json.dump(sections_data, f, indent=4)

  # pdf_path=args.out_path.replace(".json", ".pdf")
  # PDFGenerator.generate_pdf(survey_title, sections_data, output_file=pdf_path)