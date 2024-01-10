import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from pdfminer.high_level import extract_text


class LegalTextParser:
    def __init__(self, pdf_path, characters=1000):
        self.pdf_path = pdf_path
        self.characters = characters

        self.pdf_text = extract_text(self.pdf_path)

        openai.organization = os.getenv("OPENAI_ORG_ID")
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def _query(self, prompt):
        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are legal text bot, you are given a legal document and you must extract the required information from it."
                },
                {"role": "user", "content": self.pdf_text[:self.characters]},
                {"role": "user", "content": prompt},
            ],
            presence_penalty=-1,
        )

        return output["choices"][0]["message"]["content"]

    def extract_structured_data(self):
        prompt = "Extract the names of the civilian and military judge, their names are in the format: 'Hon’ble <name>'." \
                 "Remove from their names any titles and honorifics, such as Admiral, or Justice." \
                 "Determine the date, format DDMMYYYY." \
                 "Determine the name of the applicant, removing military titles and role." \
                 "Return the data in the form of a python dictionary " \
                 "{'civilian judge': <name>, 'military judge': <name>, 'date': <date>, 'applicant name': <applicant name>]."
        return self._query(prompt)


def main():
    load_dotenv()

    # Select a random PDF file from inside the judgement_pdfs directory
    # pdf_path = random.choice(list(Path('judgement_pdfs').glob('**/*.pdf')))
    pdf_path = Path('judgement_pdfs/lucknow/M.A. No. 82 & 95 of 2015 Emmual (Misc).pdf')

    # # Extract text from the PDF
    # text = extract_text(pdf_path)[:1000]
    #
    # # Count words in text
    # words = [len(line.split()) for line in text.splitlines()]
    #
    # # Print the number of words
    # print(f"Words in PDF: {sum(words)}")

    parser = LegalTextParser(pdf_path)
    print(parser.extract_structured_data())


if __name__ == '__main__':
    main()
