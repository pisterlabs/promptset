import arxiv
import datetime
from dateutil.relativedelta import relativedelta

import openai
import numpy as np
import os


from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def fetch_embed_save_papers(start_date, end_date):

    # skip if the file already exists
    if os.path.exists(f'database/papers_{start_date}_{end_date}.npy'):
        print(
            f"File papers_{start_date}_{end_date}.npy already exists, skipping...")
        return

    # Set up the search query
    search = arxiv.Search(
        query=f"(cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:cs.CL OR cat:cs.RO OR cat:cs.NE OR cat:cs.GT) AND submittedDate:[{start_date.year}{start_date.month:02d}{start_date.day:02d} TO {end_date.year}{end_date.month:02d}{end_date.day:02d}]",
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = {}

    for idx, result in enumerate(search.results()):
        try:
            papers[idx] = {}
            papers[idx]['paper_id'] = result.entry_id
            papers[idx]['abstract'] = result.summary
            papers[idx]['title'] = result.title
            papers[idx]['authors'] = result.authors
            papers[idx]['link'] = result.entry_id
            papers[idx]['pdf'] = result.pdf_url
            papers[idx]['date'] = result.published
            papers[idx]['categories'] = result.categories

            query = f"Title: {papers[idx]['title']}\n ===== Abstract: {papers[idx]['abstract']}\n\n"
            emb = embeddings.embed_query(query)

            papers[idx]['embedding'] = emb

            print('-----------------------------------')
            print(f"idx: {idx}, title: {papers[idx]['title']}")
            print('-----------------------------------')
        except Exception as e:
            print(e)
            print(f"Error with paper {idx}, skipping...")
        continue

    # save the dictionary on file:
    np.save(f'database/papers_{start_date}_{end_date}.npy', papers)


# Get embeddings and papers for the past 4 yeats, month by month
for year in range(datetime.date.today().year-12, datetime.date.today().year+1):
    for month in range(1, 13):
        start_date = datetime.date(year, month, 1)
        end_date = start_date + relativedelta(months=1)
        print(f"Fetching papers from {start_date} to {end_date}")
        fetch_embed_save_papers(start_date, end_date)
