# Import necessary libraries
import gradio as gr
import requests
import openai
openai.api_key ="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"



def text_to_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    generated_text = response.choices[0].text.strip()
    return generated_text

def text_to_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256",
    )
    generated_image_url = response["data"][0]["url"]
    return generated_image_url



# Set up Gradio interfaces
text_to_text_interface = gr.Interface(fn=text_to_text, inputs="text", outputs="text")
text_to_image_interface = gr.Interface(fn=text_to_image, inputs="text", outputs="image")

# Launch Gradio interfaces
text_to_text_interface.launch(share=True)
text_to_image_interface.launch(share=True)









