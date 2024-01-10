import openai
import gradio
import PyPDF2

openai.api_key = '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'

def process_pdf(pdf_content):
    pdf_file = PyPDF2.PdfFileReader(pdf_content)
    extracted_text = ""
    for page_num in range(pdf_file.numPages):
        page = pdf_file.getPage(page_num)
        extracted_text += page.extractText()
    return extracted_text

def CustomChatGPT(user_input, pdf_file):
    if pdf_file is not None and pdf_file.name.endswith(".pdf"):
        extracted_text = process_pdf(pdf_file.read())
        user_input += "\n" + extracted_text
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarize the uploaded document"},
            {"role": "user", "content": user_input}
        ]
    )
    
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    return ChatGPT_reply

inputs = [
    gradio.inputs.Textbox(label="User Input"),
    gradio.inputs.File(label="Upload PDF", type="file")
]

outputs = gradio.outputs.Textbox(label="Assistant Reply")

gradio.Interface(
    fn=CustomChatGPT,
    inputs=inputs,
    outputs=outputs,
    title="Upload Document - GET Any Answers !!!"
).launch(share=True)
