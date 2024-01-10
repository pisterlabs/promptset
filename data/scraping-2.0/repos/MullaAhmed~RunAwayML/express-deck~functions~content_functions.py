import openai
import pdfkit

def get_content_from_openai(prompt, engine,api_key,use_chat=False):

    openai.api_key = api_key

    if use_chat==True:
        messages = [
            {"role": "system", "content": "You are a content writer and you need to write content for a presentation."},
            {"role": "user", "content": prompt}
        ]

        # Make a request to the chat completion API
        response = openai.ChatCompletion.create(
            model=engine,
            messages=messages,
        )

        # Get the assistant's reply
        reply = response.choices[0].message['content']
    else:
        response = openai.Completion.create(
        engine=engine,  # Specify the model to use (e.g., text-davinci-003)
        prompt=prompt,
        max_tokens=3500  # Specify the maximum length of the generated text
    )
        reply=response.choices[0].text.strip()


    return reply

def render_pdfkit(html, output):

    options = {
            'page-width': '1470px',
            'page-height': '810px',
            'margin-top': '0mm',
            'margin-right': '0mm',
            'margin-bottom': '0mm',
            'margin-left': '0mm',
            # 'orientation':'Landscape'
        }

    pdfkit.from_string(html, output, options=options)