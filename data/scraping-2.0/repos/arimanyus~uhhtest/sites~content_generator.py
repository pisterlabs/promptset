import openai

def generate_content(content_type, industry, description):
    openai.api_key = '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'

    if content_type == 'about_us':
        prompt = f"Write an 'About Us' section for a {industry} company that specializes in {description}."
    elif content_type == 'home':
        prompt = f"Write a home page content for a {industry} company that specializes in {description}."
    elif content_type == 'contact':
        prompt = f"Write a contact page content for a {industry} company that specializes in {description}."
    # Add more elif conditions here for other types of content

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )

    return response.choices[0].text.strip()
