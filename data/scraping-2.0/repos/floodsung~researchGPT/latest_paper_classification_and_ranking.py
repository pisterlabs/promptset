import arxiv
import openai


def chat(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages = [{"role": "system", "content": prompt},],
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]


# find all papers mentioning "gpt" from 2023
search = arxiv.Search(
    query="GPT-4",
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

# convert search.results() to list
results = list(search.results())

papers = ""
for i in range(10):
    papers += f"paper {i+1}: {results[i].title}\n"
    papers += f"abstract: {results[i].summary}\n"
    papers += f"pdf link: {results[i].pdf_url}\n\n"



prompt = f"You are a research scientist at OpenAI studying large language model. Below is a list of latest papers and abstracts related to large language model. Please classify papers into different categories based on your knowledge and rank the papers based on its value and give some explanations.\n {papers}"
print(prompt)
print(chat(prompt))
