import openai.error
from pypdf import PdfReader
import io
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.tokenize import sent_tokenize
import nltk
from utils.openai_mgmt import num_tokens_in_string, openai_completion
from utils.consts import AI_SUMMARIZATION_TYPE


# Download Punkt Sentence Tokenizer (if not already downloaded)
nltk.download('punkt')


def get_content_from_url(url: str) -> tuple[requests.Response, str]:
    """
    Get content (text webpage or PDF) from a URL.
    """
    print(f'Getting content from URL: {url}')
    try:
        response = requests.get(url,
                                headers={
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3835.0 Safari/537.36',
                                    'Accept': '*/*'})
        content_type = response.headers['Content-Type'].lower()
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f'HTTP error occurred: {e}')
        raise
    else:
        # Check that the request was successful (status code 200)
        if response.status_code == 200:
            print(f'Received content type: {content_type}')
            return response, content_type
        else:
            raise requests.exceptions.HTTPError(f"{response.status_code}")


def retrieve_pdf_from_response(response: requests.Response, content_type: str) -> io.BytesIO:
    """
    Retrieve PDF content from response.
    param: response: response object from requests.get()
    param: content_type: content type of the response
    return: PDF file as BytesIO object
    """
    if 'application/pdf' not in content_type:
        raise ValueError(f'Invalid content type: {content_type}, expected PDF.')
    else:
        return io.BytesIO(response.content)


def extract_text_from_pdf(pdf_file: io.BytesIO) -> tuple[str, int]:
    """
    Extract text from a PDF file.
    param: pdf_file: PDF file as BytesIO object
    return: tuple (pdf_text, num_words)
    """
    reader = PdfReader(pdf_file)
    pdf_text = ''

    num_pages = len(reader.pages)
    for page in range(num_pages):
        pdf_text += reader.pages[page].extract_text()

    # remove hyphens at end of lines and connect words
    pdf_text = pdf_text.replace('-\n', '')
    # # remove newlines
    # pdf_text = pdf_text.replace('\n', ' ')

    num_words = len(nltk.word_tokenize(pdf_text))

    print(f'PDF has {num_words} words.')
    return pdf_text, num_words


def split_into_sentences(text: str, tokens_limit: int) -> list[str]:
    """
    Split a text into sentences.
    """
    list_of_sentences = sent_tokenize(text)
    resulting_list = []
    chars_limit = int(tokens_limit * 4 * 0.9)  # 4 is an average number of characters per token, we take 90% for safety

    for sentence in list_of_sentences:
        if len(sentence) > chars_limit:
            for i in range(0, len(sentence), chars_limit):
                if i + chars_limit <= len(sentence):
                    resulting_list.append(sentence[i:i + chars_limit])
                else:
                    resulting_list.append(sentence[i:])
        else:
            resulting_list.append(sentence)

    return resulting_list


def split_into_chunks(sentences: list[str], tokens_limit: int):
    """
    Split a list of sentences into chunks of text. Each chunk can have max tokens_limit tokens.
    Sentences must not be split between succeeding chunks so that the summarization is correct.
    """
    chunks = []
    current_chunk = ''
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = num_tokens_in_string(sentence)

        if sentence_tokens > tokens_limit:
            raise ValueError(f'Sentence with {sentence_tokens} tokens exceeds the limit of {tokens_limit} tokens.')

        if current_chunk_tokens + sentence_tokens > tokens_limit:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_chunk_tokens = sentence_tokens
        else:
            current_chunk += ' ' + sentence
            current_chunk_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def summarize_chunk(num_chunks: int,
                    chunk_pos: int,
                    chunk: str,
                    summarize_type: str,
                    question: str,
                    llm_model: str,
                    length_percentage: int,
                    randomness: float) -> str:
    """
    Summarize a chunk of text with a GPT model.
    """
    num_of_words = len(nltk.word_tokenize(chunk))
    short_size = int(num_of_words * length_percentage / 100)
    user_prompt = ''
    bullet_option = ''

    print('-' * 80)
    print(f'Summarizing chunk {chunk_pos} of size {num_of_words} words to max {short_size} words...\n')
    # print(f'Chunk:\n{chunk}')

    if summarize_type == AI_SUMMARIZATION_TYPE["BULLET_POINTS"]:  # 'Bullet points':
        bullet_option = 'with capturing main points and key details in form of bullets from'
    elif summarize_type == AI_SUMMARIZATION_TYPE["TEXT_SUMMARIZATION"]:  # 'Text summarization':
        bullet_option = 'with capturing main points and key details from'

    if summarize_type == AI_SUMMARIZATION_TYPE["TEXT_SUMMARIZATION"] or \
            summarize_type == AI_SUMMARIZATION_TYPE["BULLET_POINTS"]:
        user_prompt = (f"Please summarize {bullet_option} the following {chunk_pos}. part of the larger text "
                       f"in {short_size} words: "
                       f"{chunk}"
                       ) if num_chunks > 1 else (
            f"Please summarize {bullet_option} the following text in {short_size} words: "
            f"{chunk}"
        )
    elif summarize_type == AI_SUMMARIZATION_TYPE["FOCUS_QUESTION"]:
        user_prompt = (
            f"Please analyze the {chunk_pos}. part of the larger text and provide a summary in {short_size} "
            f"words focusing on the question: `{question}`. This part of the text is: "
            f"{chunk}"
            ) if num_chunks > 1 else (
            f"Please analyze the following text and provide a summary in {short_size} words focusing "
            f"on the question: `{question}`. The text is: "
            f"{chunk}"
        )

    system_prompt = ("You are a summarization expert. Your summary should be accurate and objective. "
                     "Add headings and subheadings. Use markdown for formatting.")

    gpt_prompt = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    try:
        summarization = openai_completion(gpt_prompt, llm_model, randomness)
    except openai.error.OpenAIError as e:
        raise
    else:
        return summarization


def process_text(content_text: str,
                 summarize_type: str,
                 question: str,
                 llm_model: str,
                 tokens_limit: int,
                 length_percentage: int = 20,
                 randomness: float = 0.8) -> str:
    """
    Process text to summarize it. Split into whole sentences, then make chunks with whole sentences,
    then summarize each chunk.
    """
    num_tokens = num_tokens_in_string(content_text)
    print(f'\nNumber of tokens in the content: {num_tokens}')

    sentences = split_into_sentences(content_text, tokens_limit)

    try:
        chunks = split_into_chunks(sentences, tokens_limit)
    except ValueError as e:
        raise

    num_chunks = len(chunks)

    summary_pieces = ""
    for chunk_pos, chunk in enumerate(chunks):
        try:
            summarization = summarize_chunk(num_chunks=num_chunks,
                                            chunk_pos=chunk_pos + 1,
                                            chunk=chunk,
                                            summarize_type=summarize_type,
                                            question=question,
                                            llm_model=llm_model,
                                            length_percentage=length_percentage,
                                            randomness=randomness)
        except openai.error.OpenAIError as e:
            raise
        else:
            summary_pieces = summary_pieces + '\n\n' + summarization

    return summary_pieces


def tag_visible(element) -> bool:
    """
    Filter out invisible/irrelevant parts from a webpage text.
    """
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def page_to_string(response: requests.Response, content_type: str) -> tuple[str, int]:
    """
    Get the text content of a webpage.
    """

    # response = requests.get(page_url)
    if 'text/html' not in content_type:
        raise ValueError(f'Invalid content type: {content_type}, expected HTML.')

    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove invisible/irrelevant text parts
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)

    # Join the text parts
    joined_filtered_text = u" ".join(t.strip() for t in visible_texts)
    num_words = len(nltk.word_tokenize(joined_filtered_text))

    print(f'Page has {num_words} words.')
    return joined_filtered_text, num_words
