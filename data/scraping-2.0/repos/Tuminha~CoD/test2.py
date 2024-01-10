import openai
import PyPDF2
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get API key from .env
openai.api_key = os.getenv('OPENAI_API_KEY')

def read_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Read PDF
pdf_path = "/Users/franciscoteixeirabarbosa/projects/test/CoD/doc/SSRN-id4573321.pdf"
article_text = read_pdf(pdf_path)

# Prepare the prompt
# Prepare the prompt
context = article_text[:30000]  # Adjust this number based on your needs
prompt = f"""Article: {{context}}
            You will generate increasingly concise, entity-dense summaries of the above article.

            Repeat the following 2 steps 5 times.

            Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.
            Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

            A missing entity is:
            - relevant to the main story,
            - specific yet concise (5 words or fewer),
            - novel (not in the previous summary),
            - faithful (present in the article),
            - anywhere (can be located anywhere in the article).

            Guidelines:

            - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
            - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
            - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
            - The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article.
            - Missing entities can appear anywhere in the new summary.
            - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

            Remember, use the exact same number of words for each summary.

            Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".""".format(context=context)

try:
    # Call OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000  # You can adjust this based on your needs
    )

    # Debugging: Print the entire response to see its structure
    print("Debugging: Full Response:")
    print(response)

    # Check if 'choices' key exists
    if 'choices' in response:
        # Check if 'message' and 'content' keys exist in response['choices'][0]
        if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
            print(response['choices'][0]['message']['content'].strip())
        else:
            print("Keys 'message' and/or 'content' not found in response['choices'][0].")
    else:
        print("Key 'choices' not found in response.")

except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
