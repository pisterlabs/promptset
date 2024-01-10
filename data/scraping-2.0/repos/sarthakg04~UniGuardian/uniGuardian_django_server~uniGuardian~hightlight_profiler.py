import google.generativeai as palm
from openai import OpenAI


def get_hightlight(resume_text, sop_text, lor1_text, lor2_text, api_key):
    # palm.configure(api_key = api_key)
    # models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    # model = models[0].name
    
    prompt = "get the hightlight of the applicant from the resume, SOP and letters of commenendation. Please do organize the returned text in one paragraph within 200 words!!!!:\n" + resume_text + "\n\n" + sop_text + "\n\n" + lor1_text + "\n\n" + lor2_text
    # completion_palm = palm.generate_text(
    #     model=model,
    #     prompt=prompt,
    #     temperature=0,
    #     max_output_tokens=1000,
    # )
    # print(completion_palm.result)
    # return completion_palm.result

    client = OpenAI(
        api_key = api_key,
    ) 
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ],
    )
    return completion.choices[0].message.content
