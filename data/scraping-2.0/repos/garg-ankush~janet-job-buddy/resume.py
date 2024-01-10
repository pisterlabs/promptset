import docx
import openai

def generate_resume(client_info):
    # Set up OpenAI API credentials
    openai.api_key = 'YOUR_OPENAI_API_KEY'

    # Generate resume using OpenAI API
    response = openai.Completion.create(
        engine='davinci',
        prompt='[Your Name]\n[Your Contact Information: Phone Number, Email Address]\n\n[Objective]\n...',
        max_tokens=500,
        temperature=0.6,
        n=1,
        stop=None,
        prompt_model={'prompt': client_info}
    )

    # Extract the generated resume from the API response
    resume = response.choices[0].text.strip()

    # Create a new Word document
    doc = docx.Document()

    # Add the generated resume to the document
    doc.add_paragraph(resume)

    # Save the document as a rich text format file
    doc.save('resume.docx')
