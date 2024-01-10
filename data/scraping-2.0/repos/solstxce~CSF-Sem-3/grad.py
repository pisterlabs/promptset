import gradio
# import openai
# from gradio.components import inputs
# from vars import KEY
# openai.api_key = KEY
def api_resp(option,user,token,password):
    
    return f"""The {quantity} {animal}s from {" and ".join(countries)} went to the {place} where they {" and ".join(activity_list)} until the {"morning" if morning else "night"}"""
theme='JohnSmith9982/small_and_pretty'
op=gradio.outputs.Textbox(label="API Response Text")
gradio.Radio(["park", "zoo", "road"], label="Location", info="Where did they go?")
ip=gradio.inputs.Textbox(label="Prompt Text")
demo = gradio.Interface(fn=api_resp ,inputs=[gradio.Radio(["login", "register", "api_tester","curl_it"], label="API Endpoint", info="RESTful API Endpoint")], outputs=op,theme='JohnSmith9982/small_and_pretty')
    
demo.launch()  