import os
import gradio
import openai
import cml.data_v1 as cmldata


def main():
    # Configure gradio QA app 
    print("Configuring gradio app")
    demo = gradio.Interface(fn=get_responses, 
                            inputs=[gradio.Radio(['gpt-3.5-turbo', 'gpt-4'], label="Select GPT Engine", value="gpt-3.5-turbo"), 
                                    gradio.Textbox(label="OpenAI API Key", placeholder="sk-xxxxx"), 
                                    gradio.Textbox(label="Text Question", placeholder="")],
                            outputs=[gradio.Textbox(label="Generated SQL Query")],
                            allow_flagging="never")


    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
    print("Gradio app ready")


def get_databases_and_tables():
    pass

# Helper function for generating responses for the QA app
def get_responses(engine, open_ai_api_key, question):
    if engine is "" or question is "" or open_ai_api_key is "" or engine is None or question is None or open_ai_api_key is None:
        return "No question, engine, or api key selected."

    openai.api_key = open_ai_api_key
    
    sql_schema = ""

    contextResponse = get_llm_response_with_context(question, sql_schema, engine)
    rag_response = contextResponse
    

    return rag_response

  
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llm_response_with_context(question, context, engine):
    question = "Generate a SQL query based on the context provided by the DESCRIBE TABLE output for the following: " + question
    
    response = openai.ChatCompletion.create(
        model=engine, # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": str(context)},
            {"role": "user", "content": str(question)}
            ]
    )
    return response['choices'][0]['message']['content']


if __name__ == "__main__":
    main()
