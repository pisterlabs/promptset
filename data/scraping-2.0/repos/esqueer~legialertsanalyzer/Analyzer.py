#written to run on google collab. Please make sure you start with !pip install --upgrade gspread openai PyPDF2 oauth2client

import gspread
from openai import OpenAI
from PyPDF2 import PdfReader
import requests
from google.colab import auth
from oauth2client.client import GoogleCredentials
from google.auth import default

creds, _ = default()

gc = gspread.authorize(creds)

# Set your OpenAI API key
openai.api_key = ''

# Initialize the OpenAI client
client = OpenAI(api_key=openai.api_key)

# Replace 'Your_Sheet_Name' with the actual name of your Google Sheet
sheet = gc.open('LGBTQ+ Legislative Tracking 2024').sheet1

urls = sheet.col_values(16)[1:]  # 17 corresponds to column 'P'

def extract_text_from_pdf(url):
    try:
        response = requests.get(url, timeout=10)  # Including a timeout for the request
        with open('temp.pdf', 'wb') as f:
            f.write(response.content)
        reader = PdfReader('temp.pdf')
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return ""
    except Exception as e:  # General exception to catch all PyPDF2 errors
        print(f"Error reading PDF from {url}: {e}")
        return ""

for i, url in enumerate(urls, start=2):  # start=2 assumes URLs start from row 2
    # Check if there's already a summary in column 17
    existing_summary = sheet.cell(i, 17).value
    if not existing_summary:
        pdf_text = extract_text_from_pdf(url)
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # Specify the model
            messages=[{"role": "system", "content": "You are a GPT designed to analyze anti-LGBTQ bills, with a special focus on anti-trans bills. Your primary function is to dissect these bills, identifying and highlighting provisions specifically targeting sexual orientation and gender identity. In your analysis: Use respectful and accurate terminology. Refer to medical procedures and treatments related to gender transition as 'gender-affirming care.' Avoid using outdated or derogatory terms, and replace any such terms found in the bills with 'gender-affirming care' unless quoting directly for clarity or analysis. Provide detailed analysis of how the bill functions, its implications, and potential impacts, focusing on key components like: Bans on gender-affirming care (specify if it targets minors, adults, or both). Restrictions on bathroom access based on gender identity ('trans bathroom ban'). Limitations on discussion of sexual orientation and gender identity in educational settings ('Don't Say Gay' bills). Provisions regarding 'parental rights' and their implications for LGBTQ students. Any private right of action clauses and their potential chilling effects on gender-affirming care. When discussing the terms used in the bills, provide context and clarification to ensure understanding without perpetuating harmful language. When analyzing bills related to adult-oriented performances, consider the broader context and potential implications of such legislation on the LGBTQ+ community, with a specific focus on drag performances. This includes identifying provisions that may indirectly target or affect drag performances, even if not explicitly mentioned. Assess the potential impacts these bills could have on freedom of expression and the cultural significance of drag in the LGBTQ+ community. Provide insights into how these legislative measures might be perceived as 'drag bans' and discuss their broader societal implications Contextualize within LGBTQ+ Rights: Place the bill within the broader context of LGBTQ+ rights and cultural practices, assessing how it aligns with or contradicts these principles. Highlight Potential for Misuse or Misinterpretation: Discuss how the bill might be misused or interpreted in a manner that could lead to the suppression of drag performances or the marginalization of the LGBTQ+ community. If no bill text is provided, please just reply 'N/A'"},
                    {"role": "user", "content": pdf_text}]
        )
        summary = response.choices[0].message.content  # Accessing the content using dot notation
        sheet.update_cell(i, 17, summary)  # Assuming summaries go in the 17th column
    else:
        print(f"Row {i} already has a summary, skipping.")
