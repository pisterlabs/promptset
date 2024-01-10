from langchain.embeddings.spacy_embeddings import SpacyEmbeddings
import numpy as np
from numpy.linalg import norm
import Keywords
import ResumeText
import Reparser
import cohere

co = cohere.Client('ZcJBMl4kes55aTNs8IEkHu8PSqha0PyXrAbgn5hC')
embedder = SpacyEmbeddings()

resume_number = int(input("Enter the number of resumes for this specific job description..."))
highlight_number = int(input("On a scale of 1 to 10, to what extent would you want the candidate's resume to adhere to the given job description?"))
smart_recruit = 0
include_suggestions = 0
summarizer = 0

if smart_recruit == 0:
    job_title = input("Enter the job title....")
    job_desc = input("Enter the job description...")
else:
    #Cohere API: ZcJBMl4kes55aTNs8IEkHu8PSqha0PyXrAbgn5hC
    job_title = input("Enter the job title...")
    response = co.generate(prompt= f"State the technical skills required (with emphasis on names of tools and technologies) for this job : {job_title}",)
    job_desc = response

resume_paths = [r"C:\Users\annad\OneDrive\Desktop\Dheeraj Anna.pdf", r"C:\Users\annad\OneDrive\Desktop\Dheeraj Anna.pdf", r"C:\Users\annad\OneDrive\Desktop\Dheeraj Anna.pdf", r"C:\Users\annad\OneDrive\Desktop\Dheeraj Anna.pdf"]
texts = []
skills_extracted = []
emails_extracted = []
summaries_extracted = []

for path in resume_paths:
    text = ResumeText.extract_text(path)
    texts.append(text)
    skills_extracted.append(Reparser.skills(path))

# texts = [
#     "I am doing fine.",
#     "I am doing well.",
#     "It feels really bad."
#     "I am not exactly well today!"
# ]

#keyword extraction
keywords_total = []
for i in range(0, len(texts)):
    keyword_each = Keywords.keyword_extractor(texts[i], highlight_number*2)
    string_each = "" 
    for word in keyword_each:
        string_each = string_each + word + " "
    keywords_total.append(string_each)


embeddings = embedder.embed_documents(texts)
print(embeddings)
print("\n\n\n")
embeddings_job = embedder.embed_documents([job_desc])
# print(embeddings)
print("\n\n\n")
print(embeddings_job)
print(f"The length of the embeddings for the document is {len(embeddings[0])}")
print(f"The length of the embeddings for the document is {len(embeddings_job)}")
print("\n\n\n")
# for i, embedding in enumerate(embeddings):
#     print(f"Embedding for document {i+1}: {embedding}")

# query = "I am feeling okay."
# query_embedding = embedder.embed_query(query)
# # print(f"Embedding for query: {query_embedding}")
# print(f"Embedding for query: {len(query_embedding)}")
percentage_similarities = list()
for i in range(0, len(embeddings)):
    A = np.array(embeddings[i])
    B = np.array(embeddings_job[0])
    print(f"\n\n{A} {B} {i}\n\n")
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    cosine_percent = cosine * 100
    cosine_formatted = f"{cosine_percent:.2f}"
    percentage_similarities.append(float(cosine_formatted))

#Contextual relevance percentage calculation
context_percentage_similarities = list()
embeddings_keywords = embedder.embed_documents(keywords_total)
for i in range(0, len(embeddings_keywords)):
    A = np.array(embeddings_keywords[i])
    B = np.array(embeddings_job[0])
    # print(f"\n\n{A} {B} {i}\n\n")
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    cosine_percent = cosine * 100
    cosine_formatted = f"{cosine_percent:.2f}"
    context_percentage_similarities.append(float(cosine_formatted))


print(percentage_similarities, type(percentage_similarities[0]))
print("Presenting the CV rankings for the given job description....")
# dict_cv_rank = {}
# for text_rank in range(0, len(texts)):
#     dict_cv_rank[texts[text_rank]] = [percentage_similarities[text_rank], context_percentage_similarities[text_rank]]
# percentage_similarities.sort()
# context_percentage_similarities.sort()

for elem in range(0, len(percentage_similarities)):
    print(f"CV: {texts[elem][:20]} Raw relevance: {percentage_similarities[elem]} Contextual relevance: {context_percentage_similarities[elem]}")

if include_suggestions == 1:
    for elem in range(0, len(percentage_similarities)):
        print(f"CV: {texts[elem][:20]} Raw relevance: {percentage_similarities[elem]} Contextual relevance: {context_percentage_similarities[elem]}")
        response = co.generate(prompt= f"For the job title of {job_title} I have the following skills:\n {skills_extracted[elem]}.\n\n Suggest (by merely listing, don't elaborate) the technical skills that would be required additionally for the given job title.",)
        print(f"\n\nSuggestions for the candidate:\n\n{response}")
else:
    for elem in range(0, len(percentage_similarities)):
        print(f"CV: {texts[elem][:20]} Raw relevance: {percentage_similarities[elem]} Contextual relevance: {context_percentage_similarities[elem]}")

# print("Similarity percentage between the two vectors: ", cosine_formatted)


# print(f"The similarity index between the two embeddings id {np.dot(embeddings[0], query_embedding)}.")
