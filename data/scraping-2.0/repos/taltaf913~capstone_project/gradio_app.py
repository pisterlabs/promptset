import gradio as gr
import openai


myControls = {
    "ResultControl":None,
    "Feedback":None,
    "AdditionalInfo":None
}
def predict(imageToProcess):
    #myControls["ResultContro"].value = "No Disease"
    openai.api_key = "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"
    message = "Mango Plant Diseases" 
    if message: 
        messages = []
        messages.append( 
            {"role": "user", "content": message}, 
        ) 
        chat = openai.ChatCompletion.create( 
            model="gpt-3.5-turbo", messages=messages 
        ) 
      
    reply = chat.choices[0].message.content 
    #messages.append({"role": "assistant", "content": reply})

    return ["No Disease", reply]

def submitFeedback(a,b):
    return ["User input submitted successfully"]

with gr.Blocks(allow_flagging="manual") as app :

    gr.Markdown(
    """
        # AI based plant Disease Detection Application
       
    """
    )
    imageInput = gr.Image()

    controls = []

    myControls["ResultControl"] = gr.Textbox(label='Possible Disease could be ')
    myControls["AdditionalInfo"] = gr.TextArea(label='Additional Info')
    controls.append(myControls["ResultControl"])
    controls.append(myControls["AdditionalInfo"])
    

    predictBtn = gr.Button(value='Predict')
    predictBtn.click(predict, inputs=imageInput, outputs=controls)


    gr.Markdown()
    myControls["Feedback"] = gr.Checkbox(label="Is prediction acceptable?")
    myControls["UserInput"] = gr.Textbox(label='What is the correct classification?')
    feedbackBtn = gr.Button(value='Submit Feedback')
    feedbackBtn.click(submitFeedback, inputs =[myControls["Feedback"], myControls["UserInput"]], outputs=None)




    app.launch()
