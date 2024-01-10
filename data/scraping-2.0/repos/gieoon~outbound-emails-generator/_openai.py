from openai import AsyncOpenAI
import os
import re
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Given text, generates an outbound email for people to use.
async def generate_email(title, meta_description, website_owners, company_name, my_details):
    # You are writing to a website called "{title}"
    # Their website: "{meta_description}"
    
    m = {
                "role": "user",
                "content": f"""
                    Website Title: {title}
                    Website Description: {meta_description}
                    {"Website Owners: {website_owners}" if len(website_owners) else ""} 

                    My Details: {my_details}

                    ==========================================================
                    
                    Generate a very short email to this website's owner to introduce them to my website design services.
                    Use simple and easy to read language. Keep it brief.
                    Put their company name, {company_name} in the subject and personalize the email to their company and industry, with suggested offerings based on what they might need.
                    Mention features I can provide which would be most useful to their company. 
                    Link to https://www.webbi.co.nz and let them know what value they can get from here.
                    Do not use placeholders.

                    End the email with:
                    Alex Kagaya, 
                    Founder, Webbi Digital Studio
                    https://www.webbi.co.nz
                    hello@webbi.co.nz
                    +64 022 091 0069 
                    
                    Format it as:
                    Subject: {{email subject}}
                    Content: {{email content}}
                """
            }
    print('generate_email: ', m)
    chat_completion = await client.chat.completions.create(
        messages=[
            m
        ],
        model="gpt-4-1106-preview"
    )
    
    # print('chat_completion:', chat_completion)
    return chat_completion.choices[0].message.content

async def extract_from_page (content):

    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                    Given the following content on a website, what is the owner, or contact person's name?
                    If the owner or contact person's name is not provided say <OWNERS: NONE>
                    Otherwise, say <OWNERS: {{owners}}> and replace the placeholder with the owner's name or owner's names separated by commas.

                    What is the name of the company?
                    On a new line, say <COMPANY NAME: {{company name}}> and replace the placeholder with the company's name.
                    If there is no company name, say <COMPANY NAME: NONE>

                    Website Content: {content}
                """
            }
        ],
        # model='gpt-3.5-turbo'
        model="gpt-4-1106-preview"
    )
    print('get_owners: ', chat_completion.choices[0].message.content)
    message = chat_completion.choices[0].message.content
    if '<OWNERS: NONE>' in message:
        owners = ''
        print('no owners found')

    else:
        owner_pattern = r"<OWNERS:\s*{?([^}]*)}?>"
        owner_match = re.search(owner_pattern, message)
        owners = ''
        if owner_match:
            owners = owner_match.group(1).strip()
            print("Owners found:", owners)
        else:
            print("Owners found")
    
    if '<COMPANY NAME: NONE>' in message:
        company_name = ''
    else:
        company_name_pattern = r"<COMPANY NAME:\s*{?([^}]*)}?>"
        company_name_match = re.search(company_name_pattern, message)
        company_name = ''

        if company_name_match:
            # Extract the company name
            company_name = company_name_match.group(1).strip()
            print("Company Name found:", company_name)
        else:
            print("Company name not found.")

    return [owners, company_name] 