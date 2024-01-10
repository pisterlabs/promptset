from openai import OpenAI
from googleapiclient.discovery import build

import gradio as gr

openai_args = {"api_key": ""}

sample_input = "What is the relationship between global warming and recycle?"
sample_output = "relatioship,global warming,recycle"



def get_completion(question, model="gpt-4-1106-preview"):
    prompt1 = f""" \
    The user is going to ask a question. \
    Your job is to analyze the user's question, and answer with your own knowledge. \
    If you do not know the answer to the question, you should be honest and say "I am not sure. Can you please restate the question more clearly?" \
    
    The question is delimited by triple backticks
    ```{question}```
    """

    try:
        client = OpenAI(api_key=openai_args["api_key"])
        messages1 = [{"role": "user", "content": prompt1}]
        response1 = client.chat.completions.create(
            model=model,
            messages=messages1,
            temperature=0,  # this is the degree of randomness of the model's output
        )

        textual_explanation = response1.choices[0].message.content

    
    except BaseException:
        raise gr.Error("Can't connect to GPT. Please check your OpenAI's API key.")
    
    

    return textual_explanation

def set_key(key):
    openai_args["api_key"] = key
    return

def search_youtube(question, model="gpt-4-1106-preview"):

    prompt = f""" \
    We are trying to extract key terms from the user's question, and use the extracted key terms for the input for Youtube API. \
    Your job is to analyze the user's question, and extract key terms from the user's question. \
    A key term does not have to be one word. For example, "generative AI" can be considered as one key term. \
    You should extract at most five key terms in a list form, where each key term is separated by a comma. However, if you think that the question is too ambiguous, and it does not contain any key term, \
    you should output "The question is unclear. Can you please restate the question more clearly?".

    The question is delimited by triple backticks
    ```{question}```

    Here is a sample input and output:
    sample input: {sample_input}
    sample output: {sample_output}
    """

    try:
        client = OpenAI(api_key=openai_args["api_key"])
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,  # this is the degree of randomness of the model's output
        )

        key_terms = response.choices[0].message.content

    except BaseException:
        raise gr.Error("Can't connect to GPT. Please check your OpenAI's API key.")
    

    query = key_terms.split(",")
    youtube = build('youtube', 'v3', developerKey="AIzaSyAeMxumTanwSI19nKY4KXcfp3lJIcrdQWk")
    request = youtube.search().list(q=query, part="snippet", type="video", maxResults=5)
    response = request.execute()

    # Process and format the response
    formatted_results = []
    for item in response.get("items", []):
        video_title = item["snippet"]["title"]
        video_description = item["snippet"]["description"]
        video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        
        formatted_result = f"Title: {video_title}\nURL: {video_url}\nDescription: {video_description}\n\n"
        formatted_results.append(formatted_result)


    return "\n".join(formatted_results)

def run_gradio_interface():
    with gr.Blocks() as gpt_to_youtube:
        gr.Markdown("GPT to Youtube Learning")
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    ChatGPT to Youtube Learning is a learning platform where \
                    users can ask questions by typing. This dual-source approach \
                    not only provides the learning experience for their query, \
                    but also offers both textual explanations and video content for a \
                    comprehensive understanding.

                    We employ the newest model available from OpenAI, 'gpt-4-1106-preview'. \
                    One disadvantage for this model is that it is free. \
                    With 'gpt-4-1106-preview' model, the pricing is $0.01/1,000 input tokens and \
                    $0.03/1,000 output tokens.

                    Before running the model, please submit your own OpenAI's key first. \
                    Your OpenAI's key can be generated from OpenAI's official site using your own account. \
                    We won't be able to access your key in anyway.
                    """
                )
            with gr.Column(scale=1):
                api_key = gr.Textbox(
                    label = "api_key",
                    lines = 1,
                    type = "password",
                    placeholder = "Enter your API key here",
                )

                btn_key = gr.Button(value="Submit Key")
                btn_key.click(set_key, inputs=api_key)

        gr.Markdown(
            """
            # Textual Explanation
            Get started by inputting the question into the text box below.
            
            <b>For example:</b><br/>
            What is generative AI?
            """
        )

        with gr.Row():
            with gr.Column():
                question_box = gr.Textbox(
                    label="question",
                    placeholder="Please put your question here",
                    lines=15,
                    max_lines=15,
                )

            with gr.Column():
                output_box = gr.Textbox(
                    value="",
                    label="Textual explanation",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                )

        btn = gr.Button(value="Submit")
        btn.click(
            get_completion,
            inputs=question_box,
            outputs=[output_box],
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    # Related Youtube Contents
                    
                    Using Youtube's official API, we provide at most 5 related video contents from Youtube.

                    For each video content, we provide three details: a title, URL, and description.

                    You can directly put the generated url as the address, and you will be able to see the content.

                    Please click the button below to see related Youtube Contents.
                    """
                )
            
            with gr.Column(scale=3):
                youtube_box = gr.Textbox(
                    value="",
                    label="Youtube contents",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                )


        btn_youtube = gr.Button(value="Retrieve Youtube Contents")

        btn_youtube.click(
            search_youtube,
            inputs=question_box,
            outputs=[youtube_box],
        )

    return gpt_to_youtube

