import os
import test
from langchain.embeddings import HuggingFaceInstructEmbeddings
from openai.embeddings_utils import cosine_similarity

query_embedding = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieving details from product reviews; Input: "
)

doc_embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the product review for product details and quality; Input: "
)

# load INSTRUCTOR_Transformer
# max_seq_length  512

good_review = "It ended up fitting me just fine. Overall this is a great jacket! I live in Idaho and it is a great in between jacket for cold weather. I really like the tactical look and the ability to change out the patches whenever I want (of course, you have to order those separately). Will most likely order another color! Would definitely recommend."
bad_review = "Seemed to be good quality then the first wash the zipper came apart. Terrible quality. I would not recommend this jacket."
bad_review2 = "very bad thing! dont by ever in ur life it is not good thing at all!"

good_embed = doc_embeddings.embed_query(good_review)
bad_embed = doc_embeddings.embed_query(bad_review)
bad_embed2 = doc_embeddings.embed_query(bad_review2)

# get cosine similarity using openai
print(cosine_similarity(good_embed, bad_embed))
print(cosine_similarity(good_embed, bad_embed2))
print(cosine_similarity(bad_embed, bad_embed2))

# prompt
query = "What is the quality of the jacket?"
prompt = query_embedding.embed_query(query)

# get cosine similarity between prompt and reviews.
# higher number = more relevant the review is to the query (more details)
print(cosine_similarity(prompt, good_embed))
print(cosine_similarity(prompt, bad_embed))
print(cosine_similarity(prompt, bad_embed2))


"""
chapters:
- 1
    - p1
    - p2
    - p3
- 2
    - p1
    - p2
    - p3
- 3
    - p1
    - p2
    - p3

cosine(query, c1) vs cosine(query, c2) vs cosine(query, c3)

extract top n chapters, for each:
    cosine(query, p1) vs cosine(query, p2) vs cosine(query, p3)
    extract top m paragraphs, for each:
        cosine(query, s1) vs cosine(query, s2) vs cosine(query, s3)
        extract top k sentences

combine all information into gpt model for summarization

"""
