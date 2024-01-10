from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from openai import ChatCompletion, Embedding
import dotenv

dotenv.load_dotenv()
import os
import openai
from utils import file_reader, parse_aim_resumes

openai.api_key = os.environ.get('OPENAI_API_KEY')


class LLMUtils:
    def __init__(self, model: ChatCompletion, embedding_source: FAISS, model_names: str = 'gpt-3.5-turbo'):
        self.model = model
        self.embedding_source = embedding_source
        self.similar_resumes = None  # placeholder until find_similar resumes runs
        self.model_names = model_names

    def find_similar_resumes(self, resume: str):
        """
        Use the FAISS vector db to find the resumes most similar and return the metadata of those 3 mot similar
        """
        _ = self.embedding_source.search(resume, k=100, search_type='similarity')
        _ = [i.metadata for i in _]

        # This is a placeholder since we don't have a way to retrieve actual resumes from their ids.
        # You should replace this with appropriate retrieval.
        self.similar_resumes = _

    def format_resume_examples(self, section: str) -> str:
        """
        Using only the 3 most similar resumes, format the resumes so that they create a string that looks like this:
        """
        relevant_resumes = [i for i in self.similar_resumes if
                            (i[section] != '') and
                            (i[section] != 'None') and
                            (i[section])]
        if 'gpt-4' in self.model_names:
            relevant_resumes = relevant_resumes[:1]
        elif 'gpt-3.5-turbo-16k' in self.model_names:
            relevant_resumes = relevant_resumes[:2]
        else:
            relevant_resumes = relevant_resumes[:2]
        examples = []
        for idx, resume in enumerate(relevant_resumes):
            example = f"""
Example {idx + 1}:

```resume
{resume['resume']}
```
```{section}
{resume[section]}
```
"""
            examples.append(example)
        return '\n'.join(examples) + '\nThese are examples, never include them in the extracted section. Also never use tick marks in the extracted section.'

    def extract_section(self, resume: str, section: str):
        """
        1. find similar examples for the section with FAISS
        2. prep the prompt with placeholders for examples
        3. format the examples into a string
        4. append the examples into the same string
        """
        if self.model_names == 'fine-tuned-gpt-3.5-turbo-4k':
            return self.fine_tuned_extract_section(resume, section)
        self.find_similar_resumes(resume)
        self.similar_resumes = self.format_resume_examples(section)
        system_prompt = f"""
You are a resume parser, your job is to extract information from the resume and fill a pre-determined section based on that information. 
This time, that predetermined section is the {section} section.

Here are some examples of similar resumes being parsed for the {section} section.
{self.similar_resumes}

(Do not include these examples in the extracted section. They are only for your reference. I repeat, do not include these examples in the extracted section.)
(Again, the above are for formatting reference ONLY do not include the above in the final answer)

You are writing for the {section} section. Do not return any thing other than the extracted section.
Always use human readable language. Do not use any special characters or formatting beyond bullets and paragraphs.
NEVER ADD DISCLAIMERS OR DISCLAIMERS TO THE EXTRACTED SECTION.


{'NOTE: PROFESSIONAL EXPERIENCE does NOT include education and does NOT include a list of skills. Do not inlcude these in the extracted text. These will be extracted by a different program.' if section == 'professional_experience' else ''}
{'NOTE: The label for each job title should be seperated from the date with enough tabs to put it at the end of the line. eg: "Neat Company, Remote                                                                                                                  8/2018-12/2019". That means a line 153 characters long exactly.' if section == 'professional_experience' else ''}
{'Note: Skills and Tech only includes bullet points of skills and tech and does not include work experience or project contributions' if section == 'skills_and_tech' else ''}
{'Note: Not many candidates have certifications or awards. If there are none, just return "No certifications" or "No awards". It is better to return nothing than to return a false positive.' if section == 'certifications' or section == 'awards' else ''}
{f"Note: Education is not {section}. Experience is not {section}. Open Source contributions are not {section}. Patents are not {section}. Publications are not {section}. Do not include these in the {section} section." if section in ['certifications', 'awards'] else ''}
{"Note: Certifications MUST be in the format of 'Certification Name, Issuer, Date'. Do not include anything else. They must be from a reputable source to be counted. Ignore all certifications without source. ONLY add these if you are 100% sure they belong." if section == 'certifications' else ''}
Here are some rules:
1. Do not return any thing other than the extracted section. That includes headers, footers, other sections, etc.
2. Always use • for bullets and paragraphs for paragraphs.
3. Never include disclaimers or disclaimers in the extracted section.
4. Only extract the section you are asked to extract. Do not extract other sections. (e.g. if you are asked to extract the Skills And Tech section, do not extract the Experience section) that will be taken care of by other parsers.
5. Always use • and never use - for bullets.
6. Never add commentary or intros. Only extract the section. (e.g. do not say "Here is the __ And __ section:" just return the section)
7. Never add additional sections. For example, do not add Education to the Skills And Tech section. It does not belong there. Stop when your section is complete.
8. Never include items that are not in the applicant's resume. Hallucinations are not allowed. (eg. Do not add work experience that is not in the resume)

REMEMBER: It is better to miss things than to be wrong. Always lean towards being conservative. The worst thing you can do is add experience that the candidate doesnt have.
"""
        human_prompt = f"""
REMEMBER: It is better to miss things than to be wrong. Always lean towards being conservative. The worst thing you can do is add experience that the candidate doesnt have.
Here is the resume candidates resume. Use ONLY it to generate the extracted section.
```resume
{resume}
```
"""
        with open('temp_prompt.txt', 'w') as f:
            f.write(system_prompt + '\n\n' + human_prompt)
        # gracefuly raise when the prompt is too long or the response was rejected to hitting the hard limit for billing
        try:
            response = self.model.create(
                model=self.model_names,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt},
                    {
                        "role": "user",
                        "content": human_prompt}
                ]
            )
        except openai.error.OpenAIError as e:
            if 'Request payload size exceeds the limit' in str(e):
                # Error due to prompt being too long
                raise ValueError("The prompt is too long! Try the gpt-3.5-turbo-16k model instead!") from e
            elif 'Your request was rejected because you have hit the hard limit for billing' in str(e):
                # Error due to hitting the hard billing limit
                raise ValueError("Hit the hard billing limit! Please contact your billing administrator.") from e
            else:
                # Re-raise the original error if it's another type
                raise
        print(system_prompt)
        return response['choices'][0]['message']['content']  # return the text not the completion object

    @staticmethod
    def fine_tuned_extract_section(resume, section, model='ft:gpt-3.5-turbo-0613:personal::7vbb2i7t'):
        # "ft:gpt-3.5-turbo-0613:open-humans::7uQBLVSG"
        system_prompt = f"You are a resume parser. Extract the {section} section"
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt},
                    {
                        "role": "user",
                        "content": resume}
                ],
                temperature=0.0,
            )
        except openai.error.OpenAIError as e:
            if 'Request payload size exceeds the limit' in str(e):
                # Error due to prompt being too long
                raise ValueError("The prompt is too long! Try the gpt-3.5-turbo-16k model instead!") from e
            elif 'Your request was rejected because you have hit the hard limit for billing' in str(e):
                # Error due to hitting the hard billing limit
                raise ValueError("Hit the hard billing limit! Please contact your billing administrator.") from e
            else:
                # Re-raise the original error if it's another type
                raise

        return completion['choices'][0]['message']['content']
