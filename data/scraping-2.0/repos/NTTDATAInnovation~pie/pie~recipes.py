from langchain.prompts import PromptTemplate


TAG_LIST = """[{{NAME}}, {{EMAIL}}, {{PHONE}}, {{CPR NUMBER}}, {{ORGANIZATION}}, {{LOCATION}}, {{ZIP CODE}}]"""


EXAMPLES = [
    """Text: 'Hej, jeg hedder Stephanie Pedersen. Jeg er 14 år gammel. Jeg bor med min Far på Hovedgaden 22, 2. th i Valby. Det er 2500 Valby. Mit telefonnummer er 28 28 28 28. Mit personnummer er 121212-1212'
Correct output: 'Hej, jeg hedder {{NAME}}. Jeg er 14 år gammel. Jeg bor med min Far på {{LOCATION}} i {{LOCATION}}. Det er {{LOCATION}}. Mit telefonnummer er {{PHONE}}. Mit personnummer er {{CPR NUMBER}}'""",
    """Text: 'elsker spille tennis i SIF og fodbold i KFUM. Elsker at være på twitch. Min email er cowboytoast12@gmail.com. Min far bor i Roskilde, men min adresse er hos min mor. Jeg går på Valby Lilleskole.'
Correct output: 'elsker spille tennis i {{ORGANIZATION}} og fodbold i {{ORGANIZATION}}. Elsker at være på twitch. Min email er {{EMAIL}}. Min far bor i {{LOCATION}}, men min adresse er hos min mor. Jeg går på {{LOCATION}}'.
""",
]

TEMPLATE_PROMPT = """You are a text anonymization robot. You replace PII (names, locations, phone numbers etc.) in a given Danish text with the appropriate PII tags from this list: {tag_list}

For example:

{examples}

Now, for this next text, give the correct output:

'{input_text}'
"""

# TEMPLTE_PROMPT_ENDING = (
#     "\n\nWrite the exact text in full with the PII replaced.\n"
#     + "Do not replace pronouns or nouns. "
#     + "Do not add or remove any other text. "
#     + "Do not correct grammar or spelling mistakes\n"
#     + "Text with PII replaced:\n\n'"
# )


def make_template(template=TEMPLATE_PROMPT):
    """A function to ready a template for use in the prompt"""
    return PromptTemplate.from_template(template)
