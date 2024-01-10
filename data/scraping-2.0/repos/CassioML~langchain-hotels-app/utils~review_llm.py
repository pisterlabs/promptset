from typing import List

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from utils.ai import get_llm
from utils.models import HotelReview


_linestart = "- "


def _split_bulletpoints(text: str) -> List[str]:
    """
    Parse the LLM bullet-point returned stringy blob into a list of stripped items.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bplines = [l[len(_linestart):] if l.startswith(_linestart) else l for l in lines]
    return bplines


# Calls the LLM to generate a summary of the given reviews tailored to the user's travel profile preferences.
# TODO improve the prompt. Also rename this function with a clearer name.
def summarize_reviews_for_user(
    reviews: List[HotelReview], travel_profile_summary: str
) -> str:
    summarizing_llm = get_llm()

    concatenated_reviews = "\n".join(review.body for review in reviews)

    prompt_template = """ You are an assistant helping travelers choose hotels.
    Write a bullet-point summary of the following "input reviews" for me, taking into account my travel preferences. {profile_summary}. 
    In the summary, do not include any information directly from my travel preferences. Prioritize information from the reviews that is similar to my travel preferences.    
    Do not exceed writing 4 concise bullet points. Do not include information about the person who wrote the review.
    
    Construct the summary using only the following information. INPUT REVIEWS:
    {hotel_reviews}

    EXAMPLE SUMMARY (do not use this information):
        - Good hotel, cozy and well furnished.
        - Sometimes noisy at night.
        - Some of the staff are slightly rude.
        - Weak pressure in the showers and water not always very hot.
    
    CONCISE SUMMARY: """

    query_prompt_template = PromptTemplate.from_template(prompt_template)
    populated_prompt = query_prompt_template.format(
        profile_summary=travel_profile_summary, hotel_reviews=concatenated_reviews
    )
    print(populated_prompt)

    chain = load_summarize_chain(llm=summarizing_llm, chain_type="stuff")
    docs = [Document(page_content=populated_prompt)]
    return _split_bulletpoints(chain.run(docs))


# Calls the LLM to generate a concise summary of the given reviews for a hotel.
# This is a general, base summary for the hotel and is not user-specific.
# TODO improve the prompt. Also rename this function with a clearer name.
def summarize_reviews_for_hotel(reviews: List[HotelReview]) -> str:
    summarizing_llm = get_llm()

    concatenated_reviews = "\n".join(review.body for review in reviews)

    prompt_template = """ You are an assistant helping travelers choose hotels.
        Write a bullet-point summary of the following "input reviews".
        Do not exceed writing 2 concise bullet points.

        Absolutely do not use information other than given in the "input reviews" below.

        INPUT REVIEWS:
        {hotel_reviews}

        EXAMPLE SUMMARY: 
            - Simple novel packed with meta references. 
            - Interesting characters and surprise twist at the end.

        CONCISE SUMMARY: """

    query_prompt_template = PromptTemplate.from_template(prompt_template)
    populated_prompt = query_prompt_template.format(hotel_reviews=concatenated_reviews)
    print(populated_prompt)

    chain = load_summarize_chain(llm=summarizing_llm, chain_type="stuff")
    docs = [Document(page_content=populated_prompt)]
    return _split_bulletpoints(chain.run(docs))
