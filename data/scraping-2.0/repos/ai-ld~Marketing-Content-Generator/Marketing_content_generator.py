import os
import openai
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# set up variables
api_key = 'your key'
model = 'gpt-3.5-turbo'

# set up gradio
def query(genre, channel, product_name, brand_name, descriptions, product_features, target_audience, Style, lengthen, use_emoji, num, Example):
    '''
    This function takes in the user's input and returns the inputs into a structured query,
    then connects to ChatGPT API and passes the query to ChatGPT to generat the results, and displays them on screen.
    User should fill in the input fields following the instructions.
    By clicking the SAVE button on the right, user can save the results to a csv file.
    '''
    # organize the inputs with conditional statements
    sentence = f"Write {num} {genre} for {brand_name}'s {product_name}. This {genre} is on {channel}. {descriptions}. Please emphasize on {product_features}. The {genre} should target {target_audience}. Please write in {Style} style and limit the lengthen to {lengthen} words."
    ex = f"{sentence}. Here are some examples: {Example}" if Example else sentence
    query = f"{ex} Use emoji." if use_emoji else f"{ex} Don't use emoji."
    # connect to API
    completions = openai.ChatCompletion.create(model=model, api_key=api_key, messages=[
                                               {"role": "user", "content": query}], temperature=1, top_p=1)
    # extract the needed output from the API response
    result = completions['choices'][0]['message']['content']
    return result


# set up input formats (dropdown, textbox, slider, checkbox) (parameters)
genre_dropdown = gr.inputs.Dropdown(
    ["slogan / tagline", "social media post"], label="Genre")
channel_dropdown = gr.inputs.Dropdown(["Facebook", "Instagram", "Twitter", "LinkedIn", "YouTube",
                                      "TikTok", "Pinterest", "Reddit", "Offical Website", "Blog", "Other"], label="Channel")
product_text = gr.inputs.Textbox(lines=1, label="Product / Service Name")
brand_text = gr.inputs.Textbox(lines=1, label="Brand Name")
description_text = gr.inputs.Textbox(
    lines=3, placeholder="Describe your product/service in a few sentences.", label="Descriptions")
feature_text = gr.inputs.Textbox(
    lines=3, placeholder="Please separate each feature with a comma.", label="Product / Service's Features, or the Goal of this Campaign")
ta_text = gr.inputs.Textbox(
    lines=2, placeholder="Please describe your target audience's age/sex/characteristics, etc.", label="Target Audience")
style_text = gr.inputs.Textbox(label="Conent Style / Tone")
num_bar = gr.Slider(0, 10, step=1, label="Number of Suggestions")
example_text = gr.inputs.Textbox(
    lines=3, placeholder="Optional. \nIf you have some excellent examples, please paste them here to help ChatGPT generate better suggestions.")
lengthen_text = gr.inputs.Textbox(
    label="Lengthen the Content by : ", placeholder="Enter a number.")

# set up user interface
software_name = "Marketing Content Generator"
software_desc = "This tool helps you generate marketing content for your products/services.\n\nIf you want to refine the output, just edit the input and re-submit again.\n\nPlease fill out the following information:"

demo = gr.Interface(
    fn=query,
    inputs=[genre_dropdown, channel_dropdown, product_text, brand_text,
            description_text, feature_text, ta_text, style_text, lengthen_text, "checkbox", num_bar, example_text],
    outputs=[gr.Textbox(lines=20, label="Response").style(
        show_copy_button=True)],
    title=software_name,
    description=software_desc,
    theme=gr.themes.Soft(
        primary_hue="sky",
        neutral_hue="gray"),
    allow_flagging="manual",
    flagging_options=[("SAVE 💾", "saved")],
    flagging_dir='MKTgenerator_saved',
    font_size="large"
)

# launch the interface
demo.launch(share=True, inbrowser=True)
