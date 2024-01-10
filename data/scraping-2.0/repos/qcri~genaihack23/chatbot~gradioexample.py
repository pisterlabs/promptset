import os
import dotenv
from openai import AzureOpenAI
import gradio as gr


dotenv.load_dotenv()

    
# set up Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_APIKEY"),  
    api_version=os.getenv('OPEN_API_VERSION'),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)


def produce_response(question, history):
    """
      question: user input, normally it is a question asked
      history: chat history
    """
    model=os.getenv("MODEL_NAME")

    # converting history into OpenAI message format
    messages = []
    for e in history:
      messages.append({"role": "system", "content": e[0]})
      messages.append({"role": "user", "content": e[1]})

    # adding the new message from the user
    content = """all the relavent sentences related to the question:
      Adnan was born in Qatar, and he liks football. He is a student at Qatar University.
    """
    content += question
    messages.append({"role": "user", "content": content})

    # call openAI API
    response = client.chat.completions.create(
      model=model,
      messages=messages,
    )

    # return response to user
    return response.choices[0].message.content

demo = gr.ChatInterface(
  produce_response,
  title="OpenAI Chatbot Example",
  description="A chatbot example for QCRI Generative AI Hackathon 2023",
  )

if __name__ == "__main__":
    demo.launch()
