from playwright.sync_api import sync_playwright
from playwright._impl._errors import Error as PlaywrightError
from googlesearch import search
import spacy
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer, util
import spacy

transformer_model = SentenceTransformer("all-mpnet-base-v2")  # Load transformer model
def similarity(text1, text2):
        embeddings = transformer_model.encode([text1, text2])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


nlp = spacy.load("en_core_web_sm")
def adaptive_chunking(text, min_length, max_length):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in doc.sents:
        sentence_length = len(sentence.text)
        if current_length + sentence_length > max_length or sentence_length > min_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence.text]
                current_length = sentence_length
            else:
                # Handle sentence longer than max_length
                chunks.append(sentence.text)
                current_length = 0
        else:
            current_chunk.append(sentence.text)
            current_length += sentence_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def mem_safe_adaptive_chunking(text, min_length, max_length):
    # NOTE: The parser and NER models require roughly 1GB of temporary memory per 100,000 characters in the input.
    CHUNK_SIZE = 100000
    # Split the text into smaller chunks before processing
    text_chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    adaptive_chunking_results = []
    for chunk in tqdm(text_chunks):
        adaptive_chunking_results.extend(adaptive_chunking(chunk, min_length, max_length))
    return adaptive_chunking_results




def run(playwright, res):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    all_text = ""
    
    for i in tqdm(range(len(res))):
        try:
            page.goto(res[i].url)
            print(res[i])
        
            # Get the page content
            html_content = page.content()
            
            # Assuming you have BeautifulSoup installed
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract all text
            text = soup.get_text()
            # print(text)
            # print("\n" + "#" * 100 + "\n")
            all_text += "\n" + text            
        except PlaywrightError as e:
            print(f"Failed to load page due to Playwright error: {e}. Skipping Page...")
        except TimeoutError:
            print("Failed to load page. Skipping Page...")

    browser.close()
    return all_text

if __name__ == "__main__":
    # if all_text.txt exists, read it
    if not os.path.exists("all_text.txt"):
        # Use playwright async API
        with sync_playwright() as p:
            res = list(search("Color of the apple fruit", advanced=True, num_results=20))
            all_text = run(p, res)
            with open("all_text.txt", "w") as f:
                f.write(all_text)
    with open("all_text.txt", "r") as f:
        all_text = f.read()
    print(len(all_text))
    chunks = mem_safe_adaptive_chunking(all_text, 100, 512)
    print(len(chunks))
    
    text = "The color of an apple is Red"
    sim_scores = []
    for chunk in chunks:
        sim_scores.append(similarity(text, chunk))
        
        
    # Plot the similarity scores
    import matplotlib.pyplot as plt
    plt.plot(sim_scores)
    plt.show()
    
    
    