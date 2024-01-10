# libs
import openai

# keys
from config import OPENAI_API_TOKEN

# constants

# pass the API key
openai.api_key = OPENAI_API_TOKEN

def categorize_daily_update(text):
    prompt = (
        "I need two categories. \"Daily Objectives:\" and \"Daily Accomplishments\". "
        "I am going to just start writing about my day as a ServiceNow program manager"
        " / solution architect. categorize each either sentence or "
        "phrase as an objective or an accomplishment. then complete any necessary "
        "sentence completion, add context you want. make the categorized output flow "
        "together. professionally and smartly. I submit this update with my daily time card.\n\n"
        f"Input:\n{text}\n\n"
        "Output:"
    )

    response = openai.Completion.create(
        # engine="gpt-3.5-turbo",
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

#Sample input text
#### COPY/PASTE here is a testing example below
# EXAMPLE prompt -> meet with bijah and hunter, discuss radius project methodology and first project. Had the meeting, went great, discussed coming to Charlotte next week. per Bijah's direction, I confirmed my availability and plan to be in charlotte next week. I logged in to the ServiceNow partner portal. starting to get squared away with my credentials with everything. planning to review platform implementation and other artifacts on the partner portal and now learning. reviewed design artifacts and started to organize collateral for first Radius project. met the new PM (Robin) on the call wit Hunter and Bijah. 
if __name__ == "__main__":
    print('Directions -> be specific about your day, specific about what you are working with the Client\n if there are client issues, you can log them here, never complain though.')
    print()
    objectives_accomplishments = input(f"talk at me about your objectives and accomplishments today----> ")

    formatted_output = categorize_daily_update(objectives_accomplishments)
    print(formatted_output)
    print()
    print("make sure to submit your time card and this AI generated nonsense everyday")

