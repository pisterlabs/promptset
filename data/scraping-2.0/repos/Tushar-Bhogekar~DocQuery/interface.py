import gradio as gr
import openai
from logic import set_apikey, enable_api_box, add_text, generate_response, render_file, generate_summary

# Gradio application setup
with gr.Blocks() as demo:
    # block for the title
    with gr.Row():
        gr.Text("DocQuery - Document Retrieval and Conversational AI ", style={"font-size": 24, "margin-bottom": "10px"})  # Add the title here

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(
                    placeholder='Enter OpenAI API key',
                    show_label=False,
                    interactive=True
                ).style(container=False)
            with gr.Column(scale=0.2):
                change_api_key = gr.Button('Change Key')

        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select').style(height=680)

    with gr.Row():
        with gr.Column(scale=0.70):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter"
            ).style(container=False)

        with gr.Column(scale=0.15):
            submit_btn = gr.Button('Submit')

        with gr.Column(scale=0.15):
            btn = gr.UploadButton("📁 Upload a PDF", file_types=[".pdf"]).style()


    # submitting  OpenAI API key
    api_key.submit(fn=set_apikey, inputs=[api_key], outputs=[api_key])

    # changing  API key
    change_api_key.click(fn=enable_api_box, outputs=[api_key])

    # uploading a PDF
    btn.upload(fn=render_file, inputs=[btn], outputs=[show_img])

    # submitting text and generating response
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    )

    # text summarization
    with gr.Row():
        with gr.Column(scale=0.70):
            summarization_input = gr.Textbox(
                show_label=False,
                placeholder="Enter text for summarization"
            ).style(container=False)

        with gr.Column(scale=0.15):
            summarization_btn = gr.Button('Summarize')

    # text summarization
    summarization_btn.click(
        fn=generate_summary,
        inputs=[summarization_input],
        outputs=[summarization_input]
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
