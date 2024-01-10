from nltk.tokenize import sent_tokenize
import openai
import os

def golden_summary(input, length):
    worked = True
    len2 = len(input.split())
    length = min(length, len2)
    # 100 tokens roughly 75 words. Add extra for special cases (a factor of 80)
    m_tokens = int((100 * int(length)) / 80) 
    prompt = f'''Create a perfect summary of a given text but strictly follow these rules:
       1. Dont add extra content to the input text.
       The perfect summary should only contain information in the original text.
       2. The summary should be at most {length} words long - it should preferably be less.
       The text to summarize is: 
    '''
    model = "gpt4all-falcon-q4"
    prompt += input

    print("starting the golden summary creation... might take a while")
    
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=m_tokens,
            temperature=0.20,
            top_p=0.99,
            n=1,
            echo=False,
            stream=False
        )
    except Exception as error:
        response = str(error)
        worked = False
    return response, worked

def filter_sentences_by_word_limit(text, max_word_limit):
    sentences = sent_tokenize(text)
    summarized_sentences = []
    word_count = 0  
    for sentence in sentences:
        current_words = str(sentence).split()
        if word_count + len(current_words) > max_word_limit:
            break
        summarized_sentences.append(str(sentence))
        word_count += len(current_words)
    return summarized_sentences

def response_to_sentences(response, word_limit):
    generated_text = response["choices"][0].text.strip()
    filtered = filter_sentences_by_word_limit(generated_text, word_limit)
    return filtered

def main_golden_summary(input, length, docker_manual):
    if docker_manual:
        openai.api_base = "http://localhost:4891/v1"
        openai.api_key = "not needed for a local LLM"
    else:
        # Handle communication with docker container network namespaces and host computer network namespaces
        # Connects docker container to a service running on the main host
        host = os.environ.get('HOST', 'localhost')
        openai.api_key = "not needed for a local LLM"
        openai.api_base = f"http://{host}:4891/v1"

    summary, worked = golden_summary(input, length)
    if worked:
        return response_to_sentences(summary, length)
    return str(summary)


