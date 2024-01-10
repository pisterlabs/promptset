```python
import openai
from flask import current_app as app

# Function to generate cover letter using OpenAI's GPT-3
def generate_cover_letter(job_description, company_name):
    """
    This function generates a cover letter based on the job description and company name.
    It uses OpenAI's GPT-3 model to generate the cover letter.
    """
    # Load OpenAI API key from the Flask app's configuration
    openai.api_key = app.config['OPENAI_API_KEY']

    # Define the prompt for the GPT-3 model
    prompt = f"I am applying for a job at {company_name}. The job description is as follows: {job_description}. "

    # Generate the cover letter using the GPT-3 model
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500
    )

    # Return the generated cover letter
    return response.choices[0].text.strip()
```