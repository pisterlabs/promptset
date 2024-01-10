import os
import openai

def run_conversation(user_name = "", user_address = "", user_phone = "", user_email = "", occupation_title = "", 
                     recipient_name = "", recipient_title = "", company_name = "", company_address = "",
                     past_work = "", relevant_experiences = "", excited_reasons = "", relevant_passions = "",
                     current_date = ""):
    openai.api_key = "sk-[Your Key]"

    messages = [{"role": "user", "content": 
                 f"Make a cover letter for someone named {user_name} using the following information:\n"
                 + f"Today's date is {current_date}. "
                 + f"Their email address is {user_email}, their phone number is {user_phone}, and their home address is {user_address}. "
                 + f"They're applying to be a {occupation_title} at {company_name} -- a company which is located at {company_address}. "
                 + f"They have previously worked at \"{past_work}.\""
                 + f"They have experiences in \"{relevant_experiences}.\""
                 + f"They are excited about the job because \"{excited_reasons}.\" "
                 + f"They are passionate about \"{relevant_passions}.\" "
                 + f"They're emailing {recipient_name} who has the title of {recipient_title}. "
                 + f"\nPlease include all the information given above and don't include any placeholders (i.e. [Your Name])"
    }]
    
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages)
    return response.get("choices")[0].get("message").get("content")

