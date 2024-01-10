import re
import time
from bs4 import BeautifulSoup
import re
import openai
import json

# Model to use when searching filing snippets for mentions of drug discontinuations
MODEL = "gpt-3.5-turbo-16k"

# Set of keywords associated with discontinuation
pattern = re.compile(
    r"\b(discontinue(d|s)?|halt(ed|s)?|terminate(d|s)?|stop(ped|s)?|suspend(ed|s)?|cancel(ed|s)?)\b.*?\b(drug(s)?|trial(s)?|project(s)?|research|development)",
    re.IGNORECASE,
)


class FilingAnalyzer:
    def __init__(self, db_instance, openai_api_key, openai_model=MODEL):
        openai.api_key = openai_api_key
        self._db = db_instance
        self._model = openai_model

    def _generate_prompt(self, text):
        prompt = (
            "The following text is extracted from a pharmaceutical company's SEC filing:\n'"
            + text
            + """
            Does it contain information on the discontinuation of a drug (and only a drug/compound, not a lab or other operation)?
            If so, return the following fields in valid json:

            {"discontinued": true,
            "drug_name(s)": [__],
            "reason_for_discontinuation": __}
            If the text does not refer to the discontinuation of a drug, return
            {"discontinued": false}

            Reason for discontinuation may only be present if contained within the extracted text.
            """
        )
        return prompt

    def _make_call_to_openai_api(self, match):
        """
        Takes in a snippet from the filing and adds it to the prompt for the GPT API.
        The function will return a JSON object with the discontinued drug name, if present.
        """
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._generate_prompt(match)}],
            temperature=0.6,
        )
        openai_obj = list(response.choices)[0]
        content = openai_obj.to_dict()["message"]["content"]
        output = json.loads(content)
        return output

    def _get_keyword_matches_with_context(self, filing_html, before=300, after=300):
        """Search filing for keywords associated with discontinued projects.
        The surrounding text is included for context for the GPT model to infer
        whether the discontinuation involves a drug. Full prompt must remain under the token limit for the chosen model.
        """
        soup = BeautifulSoup(filing_html, "html.parser")
        text = soup.get_text()
        last_end = 0
        for match in pattern.finditer(text):
            start = max(0, match.start() - before)
            end = min(match.end() + after, len(text))
            # If this context overlaps with the previous one, merge them
            # TODO Fix cases where this makes the prompt too long
            if start <= last_end:
                start = last_end
            last_end = end
            yield text[start:end]

    def _query_openai_and_save_response(self, match, filing_id):
        try:
            openai_result = self._make_call_to_openai_api(match)
            if openai_result["discontinued"] == True:
                for drug_name in openai_result["drug_name(s)"]:
                    if not self._db.does_compound_exist(drug_name):
                        self._db.insert_compound(
                            drug_name, filing_id, openai_result['reason_for_discontinuation'])
        except openai.error.APIError as e:
            print(f"Error: {e}. \n")

    def _find_discontinued_in_filing(self, filing_id, filing_path):
        """Find potential discontinued drugs in a given filing"""
        with open(f"{filing_path}") as f:
            filing = f.read()
            matches_with_context = list(
                self._get_keyword_matches_with_context(filing))
            print(
                f"Found {len(matches_with_context)} matches in {filing_path}")
            for i, match in enumerate(matches_with_context):
                print(f"Match {i + 1} of {len(matches_with_context)}")
                self._query_openai_and_save_response(match, filing_id)
                time.sleep(5)
            self._db.set_filing_analyzed(filing_id)

    def find_potential_drug_names_in_unprocessed_filings(self):
        """Runs parsing code on all unprocessed filings"""
        unprocessed_files = (
            self._db.get_unprocessed_filings()
        )  # get only the unprocessed files
        for (id, filing_path) in unprocessed_files:
            self._find_discontinued_in_filing(id, filing_path)
