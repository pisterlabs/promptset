import arxiv
import requests
import PyPDF2
from io import BytesIO
import tiktoken
import os
import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = os.getenv("OPENAI_API_KEY")

def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def sanitize_filename(filename):
    invalid_chars = set(r'\/:*?"<>|')
    sanitized_filename = "".join(c for c in filename if c not in invalid_chars)
    sanitized_filename = "_".join(sanitized_filename.split())
    return sanitized_filename

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_article_pdf(url):
    response = requests.get(url)
    pdf = PyPDF2.PdfReader(BytesIO(response.content))
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def sanitize_article_text(text):
    references_index = text.upper().find("REFERENCES")
    if references_index != -1:
        text = text[:references_index]
    return text

def save_article(save_path, text):
    with open(save_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)

def summarize_article(text):
    # Check the number of tokens in the text
    num_tokens = count_tokens(text)

    # Limit the text to the first 15,000 tokens if it exceeds that limit
    if num_tokens > 15000:
        text = text[:15000]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        stream=True,
        max_tokens=800,
        messages=[
            {"role": "system", "content":  "You are a high-level AI assistant capable of comprehending and summarizing complex scientific content. Your task is to digest this scientific paper and present the information in an accessible, understandable manner. Bear in mind the need to translate technical language into layman's terms wherever possible, and to prioritize the main findings, implications, and novelty of the work."},
            {"role": "user", "content": f"Here is a scientific paper that requires your expertise for short and clear summary, 5 uniqe bullet points, and the top 5 relevant keywords: {text}"}
        ]
    )

    responses = ''
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            responses += r_text
            print(r_text, end='', flush=True)
    return responses

def get_embedding(text):
    response = openai.Embedding.create(
        input=text, 
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def search_similar_articles(query, df):
    query_embedding = get_embedding(query)
    similarities = cosine_similarity([query_embedding], df["embedding"].tolist())
    top_index = similarities[0].argmax()
    return df.iloc[top_index]

def main(keyword, n, save_directory):
    print(f"MAIN! {keyword}")
    create_directory(save_directory)
    saved_filenames = set(os.listdir(save_directory))
    search = arxiv.Search(
        query=keyword,
        max_results=n,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    df_old = pd.DataFrame()
    # if csv file exists, read it in
    if os.path.exists("summary_embeddings.csv"):
        df_old = pd.read_csv("summary_embeddings.csv")

    df_new = pd.DataFrame(columns=["title", "summary", "url", "embedding"])
    for i, result in enumerate(search.results()):
        filename = sanitize_filename(result.title) + ".txt"
        print(f"TITLE: {result.title}")
        if filename in saved_filenames:
            print(f"Article {i+1} already saved.")
            continue
        text = download_article_pdf(result.pdf_url)
        # print the token count of the article
        print(f"Article {i+1} has {count_tokens(text)} tokens.")
        text = sanitize_article_text(text)
        # print the token count of the article after sanitization
        print(f"Article {i+1} has {count_tokens(text)} tokens after sanitization.")
        save_path = os.path.join(save_directory, filename)
        save_article(save_path, text)
        summary = summarize_article(text)
        embedding = get_embedding(summary)
        # append each new article to the df_new dataframe
        df_new = df_new.append({"title": result.title, "summary": summary, "url": result.entry_id, "embedding": embedding}, ignore_index=True)
        print(f"\nSummary of article {i+1}:\n{summary}\n")
        summary_filename = filename.replace(".txt", "_summary.txt")
        summary_save_path = os.path.join(save_directory, summary_filename)
        save_article(summary_save_path, summary)
    # concatenate new dataframe (df_new) with old dataframe (df_old), with new data on top
    df_combined = pd.concat([df_new, df_old], ignore_index=True)
    df_combined.to_csv("summary_embeddings.csv", index=False)



if __name__ == "__main__":
    keyword = "Brain"
    n = 100
    save_directory = "saved_articles"
    main(keyword, n, save_directory)
