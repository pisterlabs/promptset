
import openai
from key import *

openai.api_key = api_key

def evaluate_text_for_keywords(text_to_evaluate, keyword_list):
    # Define a prompt template
    prompt_template ="""
Evaluate the semantic or contextual relevance of the given textString to the list of keywords provided. Please assess the relation deeply.

Text String:
{text_to_evaluate}
Keywords:
{keyword_list}

Calculate the original strength of the textString's relation to the keywords in scale 0 to 100. Just return the scale number only.
"""

    # Create the prompt by formatting the template with provided text and keywords
    prompt = prompt_template.format(text_to_evaluate=text_to_evaluate, keyword_list=", ".join(keyword_list))

    # Make an API request to GPT-3
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,  # Adjust based on your desired response length
    )

    # Extract and return the generated response
    evaluation_result = response.choices[0].text.strip()
    return evaluation_result


if __name__ == "__main__":
    keyword_list = ['politics', 'misuse']
    text_to_evaluate = """Mayor Balen deletes controversial Facebook status
    Published On:  September 4, 2023 10:59 AM NPT By: Republica  |  @RepublicaNepal


    Mayor Balen deletes controversial Facebook status
    KATHMANDU, Sep 4: Kathmandu Metropolitan City (KMC) Mayor Balendra Shah (Balen) has deleted a controversial status he had posted on his Facebook wall on Saturday evening.

    On Saturday night, through Facebook, Mayor Shah warned to set Singha Durbar on fire after the traffic police stopped a vehicle of the KMC. He did not reveal who had stopped the vehicle and why.

    Similarly, after Shah received public criticism for making provocative statements, his secretariat issued a statement on Sunday and informed about the incident.

    The press statement issued by the mayor’s secretariat mentioned that the police stopped the vehicle carrying Mayor Shah's wife Sabina Kafle, who gave birth a few days ago, and created unnecessary hassles and treated them like criminals.

    However, Balen deleted the controversial post on Monday morning."""

    result = evaluate_text_for_keywords(text_to_evaluate, keyword_list)

    print(result)




