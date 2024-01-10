import openai
import os
from dotenv import load_dotenv
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_funding_source(sam_text):
    prompt = f"Based on your training data on NYC schools budget and allocations and this New York City School Allocation Memorandum text:\n\n{sam_text}\n\n Is this funding coming from the either the city, state, or federal government? In the begining of your response give only one word (state, federal, city, multiple, or unsure if you are unsure) then go into any further explanation"

    # Remove multiple occurrences of '\r', '\t', and '\n'
    cleaned_text = re.sub(r"[\r\t\n]+", " ", prompt)

    # Remove multiple spaces
    cleaned_text = re.sub(r" +", " ", cleaned_text)
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=cleaned_text,
            max_tokens=50,
            n=5,
            stop=None,
            temperature=0.8,
        )

    except openai.error.InvalidRequestError as e:
        return "n/a", 0, [e]

    responses = [choice.text.strip().lower() for choice in response.choices]
    funding_sources = [
        re.findall(
            r"\b(?:state|city|federal|multiple|unsure)\b", choice.text.strip().lower()
        )[0]
        for choice in response.choices
    ]
    funding_counts = {
        source: funding_sources.count(source)
        for source in ["state", "city", "federal", "multiple", "unsure"]
    }

    most_common_source = max(funding_counts, key=funding_counts.get)
    confidence = funding_counts[most_common_source] / len(funding_sources)

    return most_common_source, confidence, responses


if __name__ == "__main__":
    sam_text = """\n\n\n\n\n\n\n\n\n\n\n\n\n\nNYCDOE Division of Finance Website Navigation\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nSCHOOL ALLOCATION MEMORANDUM NO. 02, FY 2022\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDATE:  \n\n\n\n\nJune 18, 2021\n\n\n\n\n\n\n\nTO:\n\n\n\n\nExecutive Superintendents \n\n\r\n\t\t\t\t\t    \t\t        Community Superintendents \n\n\r\n\t\t\t\t\t     \t\t\tHigh School Superintendents \n\n\r\n \t\t\t\t\t     \t\t\tBorough/Citywide Office Teams \n\n\r\n \t\t\t\t\t     \t\t\tSchool Principals \n\n \n\n\n\n\n\n\n\nFROM:    \n\n\n\n\nLindsey Oates, Chief Financial Officer\n\n\n\n\n\n\n\nSUBJECT:    \n\n\n\n\nArts Supplemental Funding through Fair Student Funding\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nThis memorandum provides guidance on planning for arts programs using Fair Student Funding (FSF)\r\n and is intended to help schools effectively utilize resources and plan quality arts programming to meet NYSED arts instructional\r\n requirements for the upcoming school year.  Allocation amounts represented in this document are for advisory purposes only and\r\n have historically been provided to schools for informational purposes at their request.  Schools' FSF allocations reflect growth\r\n for increased salary costs due to collective bargaining agreements.  The change to the per capita amount for supplemental arts \r\nprogram in FY 2022 is a net increase from the FY 2021 per capita of $79.99 to $79.62. \n\n Please note that this money is already part \r\nof schools' FSF allocations.  It does not represent new funds.\n\n\n\n\n \n\n\n\n\nSchools can schedule their arts supplemental funding using \n\nTL Fair Student Funding \n\nand\r\n \n\nTL Fair Student Funding HS\n\n allocation categories.\r\n  When scheduling funds, schools should select the program description \n\nARTS SUPPLEMENT\n\n from the drop down list.  All expenditures for arts\r\n education should be noted as such in Galaxy so that we can accurately collect spending data on arts education for the annual \n\nArts in Schools Report\n\n.\n\n\n\n\nQuestions may be directed to the Office of Arts & Special Projects at\r\n\n\n\r\nArtsAndSpecialProjects@schools.nyc.gov\n\n. \r\n\r\n\n\nWith regards to the school consolidations that take effect in FY 2022, Arts Supplemental funds for the pertinent schools will be combined via the\r\n projected registers process.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDownload a copy of the School Allocation Memorandum No. 02, FY 2022\n\n\n\n\n\n\n\nAttachment:\n\n\n\n\nTable 1 - \n\nArts Supplemental Funding Summary\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nLO: ms\n\n\n\n\nC:    \n\n\n\n\nDr. Linda Chen \n\n\r\n\t\t\t\t\t     Michael Feliciano\n\n \n\n \r\n\t\t\t\t\t     Lawrence Pendergast\n\n \n\n\r\n\t\t\t\t\t     Trenton Price\n\n \n\n\r\n\t\t\t\t\t     Maria Palma\n\n \n\n\r\n\t\t\t\t\t     Lucius Young\n\n \n\n\r\n\t\t\t\t\t     Tanisha Oliver\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nNYCDOE Division of Finance Website Navigation\n\n\n\n\n\n\n\n\n"""
    funding_source, confidence, responses = get_funding_source(sam_text)
    print(f"Funding Source: {funding_source}")
    print(f"Confidence: {confidence:.2%}")

    if confidence < 0.5:
        print("Responses received:")
        for i, response in enumerate(responses, start=1):
            print(f"{i}: {response}")
