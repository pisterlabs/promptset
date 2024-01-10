import openai
import mlfiles

def process_email(body):
    apikey = mlfiles.load_setting("openai","api")
    gpt_model = mlfiles.load_setting("openai","model")
    openai.api_key = apikey
    gpt_res = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are an assistant that only speaks in markdown format"},
            {"role": "system", "content": get_email_system_prompt()},
            {"role": "user", "content": body}
        ]
    )
    tokens = str(gpt_res.usage.total_tokens)
    mlfiles.update_log("Ran a gpt request. It costed " + tokens + " tokens.")
    return gpt_res.choices[0].message.content


def get_email_system_prompt():
    text = "You will be provided an email. Analyze the email and return the following: \n Write the word 'Summary' as a Heading 1 \n Then write a summary of the important points of email in an outline format.\n Then write 'Action Items' as a Heading 2. \n Then generate a list of action items from the email. Each beginning with [] prior to the action. \n then write 'Original email' as a Heading 1. \n Finally, write the original email formatted as markdown"
    return text