import openai
from preprocess_pdf import pdf_to_text, text_to_chunks

openai.api_key = "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"


def load_recommender(path, recommender, start_page=1):
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'

def generate_text(prompt, engine="text-davinci-003"):
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_answer(question, recommender):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    prompt += "Instructions: Compose a point wise and very short and crisp reply to the query using the search results given."\
              " Don't write 'Answer:'"\
              "Directly start the answer.\n"
    
    prompt += f"Query: {question}\n\n"
    answer = generate_text(prompt)
    return answer