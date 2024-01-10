def generate_email_content(subject):
    import openai

    openai.api_key = ''  # Replace with your actual API key

    prompt = f"Compose a formal email message regarding {subject}. Avoid using placeholders like [Recipient's Name] or [Mode of Payment]."
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates formal email content."},
                {"role": "user", "content": prompt},
            ]
    )
    content = response.choices[0].message['content']
    return content



print(generate_email_content("enquire fee receipt"))
