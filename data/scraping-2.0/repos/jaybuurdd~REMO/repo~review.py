import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


def chatgpt_review(image_urls) -> str:
    # print(f"\n{url}\n")
    input = '''
        Refer to the user in second person.
        
        Review this resume based on the following criteria give detailed exampeles of any improvements you suggest based on it:

        Conciseness and Clarity: Ensure the resume is concise, clear, and focused on skills and experiences relevant to the job. Check for a brief, impactful professional summary.

        Customization: Confirm the resume is tailored to the job description, using relevant keywords and highlighting appropriate skills.

        Formatting and Consistency: Evaluate the formatting for professional appearance, consistency in font sizes and styles, and effective use of white space.

        Achievements and Metrics: Look for achievement-oriented language and quantification of achievements with metrics and statistics. Remind them to always backup metrics with tangible evidence and not just give guess numbers. Give an example on how they can do that.
        
        Active Language: Verify the use of strong action verbs and active language. Avoid passive language and personal pronouns.

        Relevance of Sections: Check for relevant sections such as work experience, education, skills, and possibly interests or projects. Ensure the most recent education and experiences are listed first.
        
        Skills Section: Assess whether the skills section is tailored to the job, including technical, language, and soft skills.
        
        Error-Free Content: Check for any spelling or grammatical errors.
        
        Professional Contact Information: Confirm the presence of up-to-date and professional contact information.
        
        International Considerations: For international applications, check if the resume adheres to country-specific norms.
        Also for any languages listed recommend they rank all the eligible one's A1 - AC based on the CEFR language scale: https://en.wikipedia.org/wiki/Common_European_Framework_of_Reference_for_Languages
        
        Avoidance of Personal Information: Ensure no unnecessary personal information is included.

        Length and Brevity: The resume should ideally be one page, especially for less than 10 years of experience.

        Make your review no more than 2-3 paragraphs.

        At the end give their resume a rating, just say: "I give your resume a X/10" don't say anything else after this.

        '''
    try:
         # Prepare the user message content with text and images
        user_content = [
            {
                "type": "text",
                "text": f"Critique this resume following these guidelines:\n\n{input}\n\nAnything else you might notice that doesn't break the guidelines you can mention as well."
            }
        ]

        # Append image URLs
        user_content.extend(
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        )

        # Create the messages list for the API call
        messages = [
            {
                "role": "system",
                "content": "You review resumes and only provide improvements that need to be made to them."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=4096
        )

        message = response.choices[0].message
        content = message.content
            
        # print(f"\n\n {content}")
        # print(f"finish: {response.choices[0].finish_reason}")
        return content
    except Exception as e:
        print(f"Error processing review: {e}")
        raise(e)

    # def chatgpt_review(text) -> str:
    #     print(f"\n{text}\n")
    #     response = client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[
    #             {"role": "system", 
    #             "content": "You review resume's and only provide improvements that need to be made to them."},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": f"Critique this resume this resume:\n\n{text}"},
    #                 ],
    #             }
    #         ],
    #     )

    #     message = response.choices[0].message
    #     content = message.content
        
    #     print(f"\n\n {content}")
    #     return content
