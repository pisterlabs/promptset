import os
import openai
import re
from config import API_KEY

openai.api_key = API_KEY

def generate_report(transcript, call_type):

    # print(f"Classified as {call_type}")


    prompt_response = []

    # Define parameters for each call type
    parameters = {
        "inboundcancel": """
        1. Did the agent mention his/her name over the call?" if yes then true else false.
        2. “For existing customers, did the agent confirm their account number or Privacy check or Account assessment?” if yes then true else false.
        3. Did the agent close the call in an appropriate manner?" if yes then true else false.
        """,

        "inboundtransfersale": """
        Section 1: Call Opening
        1. "Did the agent mention his/her name over the call?", if yes then true else false.
        2. "Did the agent inquire about RACT Number (Tasmania only)?" if yes then true else false.

        Section 2: Compliance
        3. “Did the agent present the product accordingly?" if yes then true else false.
        4. “Did the Sales Agent tailored his approach when the customer clearly indicated that he/she is after information only and offered the customer requested with the option of follow up in a few days to help with comparisons and selecting the right electricity/gas plan?" if yes then true else false.
        5. “Did the Agent clearly explain that Welcome Pack is a contract binding document?” if yes then true else false.
        6. “Did the agent advise and did the customer clearly understand about the cooling off period?" if yes then true else false.
        7. “Did the agent give the customer a conscionable decision?" if yes then true else false.
        8. Did the agent identify customer vulnerability and take appropriate steps?
         “Customer sounds aloof (not engaged)" if yes then true else false.
        and “Confused" if yes then true else false.
        and “hearing issues" if yes then true else false.
        and “repeating themselves" if yes then true else false.
        and “questioning continuously what the call is all about" if yes then true else false.
        9. “Did the agent use potential misleading or deceptive statements?" if not then true else false.

        Section 3: Authorized contact
        10. "Did the customer indicate/state they are not authorized to conduct the transfer?", if yes then true else false.

        Section 3: Obtaining and Presenting Skills
        11. "Did the agent confirm Life Support?" if yes then true else false.
        12. "Did the agent confirm the NMI/MIRN and/or supply address?" if yes then true else false.
        13. "Did the agent confirm Concession?" if yes then true else false.
        14. "Did the agent offer to read out the rates?" if yes then true else false.
        15. "DDid Agent Offer to do Bill Comparison?" if yes then true else false.
        16. "Did the agent provide an estimation for low, medium, and high usage in VIC?" if yes then true else false.
        17. "Did the agent answer customer questions correctly?" if yes then true else false.

        Section 4: Soft Skills
        18. "Did the agent avoid long silences during the call?" if yes then true else false.
        19. "Did the agent ensure they did not interrupt the customer while they spoke?" if yes then true else false.
        20. "Did the agent display a professional manner throughout the call?" if yes then true else false.
        21. "Did the agent proactively add value throughout the call?" if yes then true else false.
        22. "Did the agent sound clear and confident throughout the call?" if yes then true else false.
        23. "Did the agent refrain from using jargon during the call?" if yes then true else false.
        24. "Did the agent sound friendly, polite, and welcoming?" if yes then true else false.
        25. 'Did the agent use effective questioning skills?" if yes then true else false.
        26. "Did the agent demonstrate active listening?" if yes then true else false.
        27. "Did the agent adapt to the customer?" if yes then true else false.

        Section 5: End Call
        28. "Did the agent ask the customer, 'Do you have any other questions while I have you'? " if yes then true else false.
        29. "Did the agent close the call in an appropriate manner?" if yes then true else false.
        """,

        "inboundmoveinsale": """
        Section 1: Call Opening
        1. "Did the agent mention his/her name over the call?", if yes then true else false.
        2. For existing customers, did the agent confirm:
        Account number and Privacy check?
        and Account assessment?" if yes then true else false.
        3. "Did the agent inquire about RACT Number (Tasmania only)?" if yes then true else false.

        Section 2: Compliance
        4. "Did the agent provide information about the Move-In timeframe?" if yes then true else false.
        5. "Did the agent disclose Move-In fees?" if yes then true else false.
        6. "Did the agent disclose exceptions to associated fees (if applicable)?" if yes then true else false.
        7. "Did the agent present the product accordingly?" if yes then true else false.
        8. "Did the agent tailor their approach when the customer indicated they were seeking information only, offering follow-up assistance for comparisons and selecting the right electricity/gas plan?" if yes then true else false.
        9. "Did the agent provide the customer with a conscionable decision?" if yes then true else false.
        10. "Did the agent clearly explain that the Welcome Pack is a contract-binding document?" if yes then true else false.
        11. "Did the agent advise the customer about the cooling-off period, and did the customer clearly understand it?" if yes then true else false.
        12. "Did the agent identify customer vulnerability and take appropriate steps, considering factors such as the customer sounding aloof, confused, having hearing issues, repeating themselves, or continuously questioning the purpose of the call?" if yes then true else false.
        13. "Did the agent use any potentially misleading or deceptive statements?" if no then true else false.

        Section 3: Obtaining and Presenting Skills
        14. "Did the agent confirm Life Support?" if yes then true else false.
        15. "Did the agent confirm the NMI/MIRN and/or supply address?" if yes then true else false.
        16. "Did the agent confirm Concession?" if yes then true else false.
        17. "Did the agent offer to read out the rates?" if yes then true else false.
        18. "Did the agent provide an estimation for low, medium, and high usage in VIC?" if yes then true else false.
        19. "Did the agent answer customer questions correctly?" if yes then true else false.

        Section 4: Soft Skills
        20. "Did the agent avoid long silences during the call?" if yes then true else false.
        21. "Did the agent ensure they did not interrupt the customer while they spoke?
        22. "Did the agent display a professional manner throughout the call?" if yes then true else false.
        23. "Did the agent proactively add value throughout the call?" if yes then true else false.
        24. "Did the agent sound clear and confident throughout the call?" if yes then true else false.
        25. "Did the agent refrain from using jargon during the call?" if yes then true else false.
        26. "Did the agent sound friendly, polite, and welcoming?" if yes then true else false.
        27. 'Did the agent use effective questioning skills?" if yes then true else false.
        28. "Did the agent demonstrate active listening?" if yes then true else false.
        29. "Did the agent adapt to the customer?" if yes then true else false.

        Section 5: End Call
        30. "Did the agent ask the customer, 'Do you have any other questions while I have you'? " if yes then true else false.
        31. "Did the agent close the call in an appropriate manner?" if yes then true else false.
        """,

        "inboundretention": """
        Section 1: Call Opening
        1. "Did the agent mention his/her name over the call?", if yes then true else false.

        Section 2: Compliance
        2. “Did the agent present the product accordingly?" if yes then true else false.
        3. “Did the Sales Agent tailored his approach when the customer clearly indicated that he/she is after information only and offered the customer requested with the option of follow up in a few days to help with comparisons and selecting the right electricity/gas plan?" if yes then true else false.
        4. “Did the agent advice and did the customer clearly understand about the cooling off period?" if yes then true else false.
        5. “Did the agent give the customer a conscionable decision?" if yes then true else false.
        6. Did the agent identify customer vulnerability and take appropriate steps?
        “Customer sounds aloof (not engaged)" if yes then true else false.
        and “Confused" if yes then true else false.
        and “hearing issues" if yes then true else false.
        and “repeating themselves" if yes then true else false.
        and “questioning continuously what the call is all about" if yes then true else false.
        7. “Did the agent use potential misleading or deceptive statements?" if not then true else false.

        Section 3: Authorized contact
        8. "Is the customer a primary account holder?", if yes then true else false.

        Section 4: Obtaining and Presenting Skills
        14. "Did the agent confirm Life Support?" if yes then true else false.
        15. Did the agent confirm customer details:
        “Account Number" if yes then true else false.
        and “Full Name" if yes then true else false.
        and “Date of Birth" if yes then true else false.
        and “Address" if yes then true else false.
        and “NMI" if yes then true else false.
        16. "Did the agent confirm Concession?" if yes then true else false.
        17. "Did the agent offer to read out the rates?" if yes then true else false.
        18. "Did Agent offer to show benefits of the retention offer against competitors?" if yes then true else false.
        19. "Did the agent answer customer questions correctly?" if yes then true else false.

        Section 5: Soft Skills
        20. "Did the agent avoid long silences during the call?" if yes then true else false.
        21. "Did the agent ensure they did not interrupt the customer while they spoke?" if yes then true else false.
        22. "Did the agent display a professional manner throughout the call?" if yes then true else false.
        23. "Did the agent proactively add value throughout the call?" if yes then true else false.
        24. "Did the agent sound clear and confident throughout the call?" if yes then true else false.
        25. "Did the agent refrain from using jargon during the call?" if yes then true else false.
        26. "Did the agent sound friendly, polite, and welcoming?" if yes then true else false.
        27. 'Did the agent use effective questioning skills?" if yes then true else false.
        28. "Did the agent demonstrate active listening?" if yes then true else false.
        29. "Did the agent adapt to the customer?" if yes then true else false.

        Section 6: End Call
        30. "Did the agent ask the customer, 'Do you have any other questions while I have you'? " if yes then true else false.
        31. "Did the agent close the call in an appropriate manner?" if yes then true else false.

        """,

        "inboundsmssale": """
        Section 1: Call Opening
        1. "Did the agent mention his/her name over the call?", if yes then true else false.

        Section 2: Compliance
        2. “Did the agent present the product accordingly?" if yes then true else false.
        3. “Did the Sales Agent tailored his approach when the customer clearly indicated that he/she is after information only and offered the customer requested with the option of follow up in a few days to help with comparisons and selecting the right electricity/gas plan?" if yes then true else false.
        4. “Did the agent advice and did the customer clearly understand about the cooling off period?" if yes then true else false.
        5. “Did the agent give the customer a conscionable decision?" if yes then true else false.
        6. Did the agent identify customer vulnerability and take appropriate steps?
        “Customer sounds aloof (not engaged)" if yes then true else false.
        and “Confused" if yes then true else false.
        and “hearing issues" if yes then true else false.
        and “repeating themselves" if yes then true else false.
        and “questioning continuously what the call is all about" if yes then true else false.
        7. “Did the agent use potential misleading or deceptive statements?" if not then true else false.

        Section 3: Authorized contact
        8. "Is the customer a primary account holder?", if yes then true else false.

        Section 4: Obtaining and Presenting Skills
        14. "Did the agent confirm Life Support?" if yes then true else false.
        15. Did the agent confirm customer details:
         “Account Number" if yes then true else false.
        and “Full Name" if yes then true else false.
        and “Date of Birth" if yes then true else false.
        and “Address" if yes then true else false.
        and “NMI" if yes then true else false.
        16. "Did the agent confirm Concession?" if yes then true else false.
        17. "Did the agent offer to read out the rates?" if yes then true else false.
        18. "Did Agent offer to show benefits of the retention offer against competitors?" if yes then true else false.
        19. "Did the agent answer customer questions correctly?" if yes then true else false.

        Section 5: Soft Skills
        20. "Did the agent avoid long silences during the call?" if yes then true else false.
        21. "Did the agent ensure they did not interrupt the customer while they spoke?" if yes then true else false.
        22. "Did the agent display a professional manner throughout the call?" if yes then true else false.
        23. "Did the agent proactively add value throughout the call?" if yes then true else false.
        24. "Did the agent sound clear and confident throughout the call?" if yes then true else false.
        25. "Did the agent refrain from using jargon during the call?" if yes then true else false.
        26. "Did the agent sound friendly, polite, and welcoming?" if yes then true else false.
        27. 'Did the agent use effective questioning skills?" if yes then true else false.
        28. "Did the agent demonstrate active listening?" if yes then true else false.
        29. "Did the agent adapt to the customer?" if yes then true else false.

        Section 6: End Call
        30. "Did the agent ask the customer, 'Do you have any other questions while I have you'? " if yes then true else false.
        31. "Did the agent close the call in an appropriate manner?" if yes then true else false.

        """,


        #### Outbound params goes here

        ### Channel params goes here

        "creditqualityassurance": """

        """,


        "example": """
            Call Opening: 30 points
            ID Check: 30 points
            Clear Communication: 40 points
        """,
        # TODO: Parameters for rest of call types
    }

    # Retrieve parameters based on call type
    params = parameters.get(call_type, "")

    # print(params)

    # print(params)


    messages = [
        {
            "role": "system", 
            "content": f"As a Quality Assurance Analyst, evaluate the provided CSR call transcript based on the following parameters for each section. check which parameters for each section is true and print the assigned number to the parameters.'"
        },
        {"role": "user", "content": f"""Here is the transcript:'{transcript}
                                    Understand the transcript first. Based on the content and context of the conversation, which parameters are true?
                                    Here are the parameters:{params}
                                    Give me the list of serial number of parameters along with True or False along with the reason of the choice
                                    """
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=650,
    )

    return response["choices"][0]["message"]['content'].strip()

    # return extract_numbers(response["choices"][0]["message"]['content'].strip())


# def extract_numbers(string):
#     # Extract the numbers using regex pattern
#     numbers = re.findall(r'\b\d+[\w.]*\b', string)
#     numbers = [num.replace('.', '') for num in numbers]
#     return numbers

#     # final_response = ""
#     # for response in prompt_response:
#     #     final_response += response

#     # return final_response

# Example usage

# file_path = 'transcripts/retention_2.txt'
# file = open(file_path, 'r')

# transcript = file.read()

# call_type = "inboundretention"

# report = generate_report(transcript, call_type)
# print(report)
