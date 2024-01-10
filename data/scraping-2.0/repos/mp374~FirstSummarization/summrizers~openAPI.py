import time

from data_set_managers.read_data_set import get_list_all_stories_and_summaries

key = "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"

import openai

# Set up your OpenAI API key
openai.api_key = "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"

source_text = "reception staff incredibly rude generally helpful told wed speak however got obnoxious receptionist " \
              "speaking clinician fob u send u hospital despite emergency finally case see following day turning " \
              "ensure got day appointment never get phone u emergency room would appropriate told collect stool " \
              "sample handed reception team however informed sample drop time drop back u end day subsequently " \
              "collection drop subsequently told wed take another next day collection month old daughter. day " \
              "appointment receptionist got last one available see nurse practitioner surgery chance advice despite " \
              "short staffed due illness yes human care given reception nurse practitioner saw later day examining " \
              "taking history medication appreciative chance feeling well track follow dont improve pharmacy work " \
              "small space medication always found surgery need medical help understand part process though use " \
              "system order good time manage appointment necessary go ae walk information good medical advice simple " \
              "emergency feel patient responsibility understand help part process awful behaviour waiting room " \
              "towards unrealistic staff job triage appropriately well done ley hill surgery pharmacy despite " \
              "difficult time thank. used great booking service get see doctor might wait restricted surgery morning " \
              "open get appointment trouble lot people exactly time get speak someone told left call back next day " \
              "sadly impression open purely convenience staff employed. easy book appointment slot next working day " \
              "minimal wait time friendly made feel ease ensure date round good service "

# Construct the prompt
prompt = f"generate a title:\n{source_text}\n\n"

response = openai.Completion.create(
    engine="text-davinci-001",  # Choose the GPT-3 engine
    prompt=prompt,
    max_tokens=10,  # Specify the maximum length of the summary
    temperature=0.7  # Adjust the temperature for creativity vs. consistency
)
summary = response.choices[0].text.strip()
print("Generated summary: ", summary)

def get_summaries():
    stories, summaries = get_list_all_stories_and_summaries()
    generated_summaries = []
    prompts = []
    for item in zip(stories, summaries):
        # Source text for summarization
        source_text = item[0]

        # Construct the prompt
        prompt = f"generate a title:\n{source_text}\n\n"

        prompts.append(prompt)

    for i in (0, 20, 40, 60, 80):
        # Generate summary using GPT-3
        response = openai.Completion.create(
            engine="text-davinci-001",  # Choose the GPT-3 engine
            prompt=prompts[i:i+20],
            max_tokens=10,  # Specify the maximum length of the summary
            temperature=0.7  # Adjust the temperature for creativity vs. consistency
        )
        generated_summaries.extend(choice.text.strip() for choice in response.choices)
        print("one set done")
        time.sleep(20)

    return summaries, generated_summaries
