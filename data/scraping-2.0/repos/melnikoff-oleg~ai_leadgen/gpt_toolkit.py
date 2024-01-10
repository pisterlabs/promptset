import json
from typing import Optional
import openai


def add_spending(model: str, spendings_counter: dict) -> dict:
    spendings_counter["openai"][model]["requests count"] += 1
    spendings_counter["openai"][model]["money spent $"] += spendings_counter["openai"][
        model
    ]["price per request $"]


def autocomplete(
    prompt: str, spendings_counter: dict, model: str = "gpt-3.5-turbo"
) -> str:
    another_model = "gpt-4" if model == "gpt-3.5-turbo" else "gpt-3.5-turbo"
    try:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.8,
        )
        add_spending(model, spendings_counter)
    except:
        completion = openai.ChatCompletion.create(
            model=another_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.8,
        )
        add_spending(another_model, spendings_counter)
    return completion.choices[0].message.content


def get_linkedin_profile_summary(
    profile: Optional[dict], spendings_counter: dict
) -> Optional[str]:
    if profile is None:
        return None
    try:
        work_experience = ""
        for i, job in enumerate(profile["work"]):
            work_experience += f"Role {i+1}:{job}\n"
        prompt = f"""Here is a LinkedIn profile of a person. Please write a short summary of his career path.
    Name: {profile['name']}
    Headline: {profile['headline']}
    Description: {profile['description']}
    Work experience from the latest to the earliest:
    {work_experience}
    Write a summary in the bullet format of this person's career path (ONLY 10 SENTENCES MAXIMUM), include notable and unusual recent facts about him:
    """
        # print(prompt, "\n")
        return autocomplete(prompt, spendings_counter)
    except:
        print("Something went wrong with this profile", profile)
        return None


def get_linkedin_company_summary(
    company: Optional[dict], spendings_counter: dict
) -> Optional[str]:
    if company is None:
        return None
    try:
        prompt = f"""Here is a LinkedIn profile of a company. Please write a short summary of their services.
    Name: {company['name']}
    Headline: {company['headline']}
    Description: {company['description']}
    Write a summary in the bullet format about their services (ONLY 10 SENTENCES MAXIMUM), define their industry and service precisely, define their ideal target customer, include notable and unusual facts about their company:
    """
        # print(prompt, "\n")
        return autocomplete(prompt, spendings_counter)
    except:
        print("Something went wrong with this profile", company)
        return None


def get_marketing_letter(
    personal_summmary: Optional[str],
    company_summmary: Optional[str],
    spendings_counter: dict,
) -> Optional[str]:
    if personal_summmary is None or company_summmary is None:
        return None
    try:
        prompt = f"""You're a THE BEST marketing and cold B2B outreach expert ON EARTH. Your task is to write an email letter promoting B2B lead generation service for a specific B2B company.
        REQUIREMENTS:
        - ALL EMAIL SHOULD BE HYPER PERSONALIZED
        - mention the company's industry and service precisely, define their ideal target customer, include notable and unusual facts about their company
        - write a personalised subject line (HARD LIMIT 50 CHARACTERS) that stands out and makes the recipient want to open the email. DO YOUR BEST HERE
        - write a short email body (max 100 words) that makes the recipient want to reply
        - use AIDA framework (Attention, Interest, Desire, Action) to structure your email
        - NEVER BE FAKE, OR CHEESY, DON'T SAY YOU LIKE THEIR WORK, OR THAT YOU'RE FOLLOWING THEM OTHERWISE I'LL KILL YOU
        - in the end of email ask if I can send through some more info
        - in the end write 'Kind regards, Oleg Melnikov | Evolva.ai'
        - don't mention evolva ai in the email body
        - don't use complex english words, keep in mind that the recipient may be a non-native english speaker
        - add new lines so it would be very easy to read, and text should look like mountain peaks - short sentences interspersed with long ones
        What exact services that you're selling:
        - booking meetings with qualified potential clients for B2B businesses
        - mention which specific type of clients we can bring to them (their ideal profile)
        - we use cold email for it, but we do it in a very personalized way using AI, that is our secret sauce, we stand out and can get a response from a potential client even if it's a big company and they get 100s of emails per day
        - address why we are better: instead of sending general offering, we will address specific needs and pain points of the concrete person that we're reaching out to. PROVIDE EXAMPLE that is clear to them
        - we work on performance basis, so we charge only for actual meetings with qualified leads
        - we provide 3 meetings with qualified leads, and if you don't like them - you pay nothing
        - MAKE SURE THEY UNDERSTAND OUR SERVICE CLEARLY
        - we do all the hard work: set up email infrastructure, train AI to generate personalized emails, write emails, send them, follow up, book meetings, and provide you with a dashboard where you can see all the campaign stats
        Summary of the person that we're reaching out to:
        {personal_summmary}
        Summary of the company that we're reaching out to:
        {company_summmary}
        Write an email letter according to the REQUIREMENTS above, also add TWO follow-up letters WITHOUT a subject line, make it short (HARD LIMIT 25 WORDS), BE CREATIVE and refer to their industry in every follow-up AND ADD NEW INFORMATION ABOUT OUR SERVICE OR USE A DIFFERENT ANGLE:"""
        # print(prompt, "\n")
        return autocomplete(prompt, spendings_counter, "gpt-4")
    except:
        print("Something went wrong with this profile", personal_summmary)
        return None


def get_json_packaged_marketing_letters(
    letters: Optional[str], spendings_counter: dict
) -> Optional[dict]:
    if letters is None:
        return None
    try:
        prompt = f"""Take given text and package it into a JSON file in the following format:
    {{
        "subject": "Subject line of the email",
        "body": "Hi Name, 
        ....
        ...",
        "follow_up_1": "First follow-up email
        ....",
        "follow_up_2": "Second follow-up email
        ..."
    }}
    Given text:
    {letters}
    JSON file:
    """
        # print(prompt, "\n")
        return json.loads(autocomplete(prompt, spendings_counter))
    except:
        print("Something went wrong with this profile", letters)
        return None
