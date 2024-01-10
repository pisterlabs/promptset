import openai
import gradio as gr
from utils.langchain_helper import init_embedding, read_split_doc, create_db_from_documents, init_llm_qa_chain

# ----------------------------------------------------------------------------
# Gradio Interface
def chat_to_your_data_ui(openai_api_key, doc_type, doc_path, chunk_size, chunk_overlap,
                         llm_name, temperature, share_gradio, image_path):    
    # ----------------------------------------------------------------------------
    # Interface functionality
    # ------------------------
    # When set OpenAI API key : read from text box
    def read_key_from_textbox(openai_api_key):    
        try: 
            from utils.openai_helper import get_completion
            openai.api_key = openai_api_key    
            response = get_completion("test", model='gpt-3.5-turbo')
            return "OpenAI API key set!"            
        except:
            return "OpenAI API key not valid!"
    
    # ------------------------
    # When reading the document
    def reading_doc_msg(doc_type, doc_path):
        return f"Reading document {doc_path} of type {doc_type} ..."
    def read_doc_msg():
        return "Finished reading the document! Let's chat!"
    def clear_chatbot_after_read_doc():            
        return "", ""
    # -------------------------
    # Init the LLM and read document
    def init_read_doc(doc_type, doc_path, chunk_size, chunk_overlap, temperature):
        global qa_chain
        # Init embedding
        embedding = init_embedding(openai.api_key)

        # Read and split document using langchain
        print(f"Reading document {doc_path} of type {doc_type} ...")
        docs_split = read_split_doc(doc_type, doc_path, chunk_size, chunk_overlap)
        # -------------------------
        # Create vector database from data    
        db = create_db_from_documents(docs_split, embedding)
        # -------------------------
        # Init the LLM and qa chain
        llm, qa_chain, memory = init_llm_qa_chain(llm_name, temperature, openai.api_key, db)            

    # When question 
    def qa_input_msg_history(question, chat_history):
        # QA function that inputs the answer and the history.
        # History managed internally by ChatInterface       
        answer = qa_chain({"question": question})['answer']
        #response = qa_chain({"question": input})
        chat_history.append((question, answer))
        return "", chat_history    
    
    # When clear all (OpenAI API key, document, chatbot)
    def clear_all():
        global qa_chain, db
        openai.api_key = None
        qa_chain = None
        db = None
        return "OpenAI API key cleared!", "Document cleared!", "", "", "", ""

    # ----------------------------------------------------------------------------
    # UI
    with gr.Blocks(theme=gr.themes.Glass()) as demo:
        # Description            
        gr.Markdown(
        """
        # Chat to your data
        Ask questions to the chatbot about your document. The chatbot will find the answer to your question. 
        You can modify the document type and provide its path/link. 
        You may also modify some of the advanced options.     

        """)
        # -------------------------
        # OpenAI API key (if not provided)
        if openai_api_key is None:
            gr.Markdown(
            """
            ## Provide OpenAI API key   
            You need to provide an OpenAI API key to use the chatbot. You can create an account and get a key [here](https://platform.openai.com/docs/api-reference/authentication/).  
            **Delete the key after using the chatbot !!!** (this will set openai.api_key=None) 
            """, scale=1
            )
            with gr.Row():
                text_openai_api_key = gr.Textbox(label="OpenAI API key", placeholder="Provide OpenAI API key!", scale=4)
                btn_openai_api_key = gr.Button("Set OpenAI API key", scale=1)
                text_openai_api_key_output = gr.Textbox(label="Reading state", interactive=False, 
                                              placeholder="OpenAI API key not provided!", scale=2)
            # -------------------------    
            # When set OpenAI API key : read from text box
            btn_openai_api_key.click(read_key_from_textbox, 
                                    inputs=text_openai_api_key,
                                    outputs=text_openai_api_key_output, 
                                    queue=False)
        # -------------------------
        # Parameters and chatbot image
        with gr.Row():
            with gr.Column(scale=2):
                # -------------------------
                # Parameters
                # Temperature and document type
                gr.Markdown(
                """
                ## Select parameters
                Default parameters are already provided.
                """
                )
                # Advanced parameters (hidden)
                with gr.Accordion(label="Advanced options",open=False):
                    gr.Markdown(
                    """
                    The document is split into chunks, keeping semantically related pieces together and with some overlap. 
                    You can modify the chunk size and overlap. The temperature is used to control the randomness of the output 
                    (the lower the temperature the more deterministic the ouput, the higher its value the more random the result, with $temperature\in[0,1]$).
                    """
                    )        
                    sl_temperature = gr.Slider(minimum=0.0, maximum=1.0, value=temperature, label="Temperature", 
                                                scale=2)
                    with gr.Row():
                        num_chunk_size = gr.Number(value=chunk_size, label="Chunk size", scale=1)
                        num_chunk_overlap = gr.Number(value=chunk_overlap, label="Chunk overlap", scale=1)


            # Chatbot image
            # https://drive.google.com/file/d/1HDnBsdfUYrCHOFtP2-DqomcmBSs9XyNI/view?usp=sharing
            # ![](https://drive.google.com/uc?id=1HDnBsdfUYrCHOFtP2-DqomcmBSs9XyNI)
            gr.Markdown(
            f"""
            <img src="{image_path}" alt="drawing" width="300"/>            
            """, scale=1)

        # -------------------------
        # Select and read document
        gr.Markdown(
        """
        ## Select document
        Select the document type and provide its path/link (eg. https://en.wikipedia.org/wiki/Lyon).
        """)
        with gr.Row():
            drop_type = gr.Dropdown(["url", "pdf", "youtube"], 
                                    label="Document Type", value=doc_type, min_width=30, scale=1)
            text_path = gr.Textbox(label="Document Path/URL", placeholder=doc_path, scale=5)
        
        with gr.Row():
            # Read document
            btn_read = gr.Button("Read document")
            text_read_output = gr.Textbox(label="Reading state", interactive=False, placeholder="Select document type and path!")

        # -------------------------
        # Chatbot
        gr.Markdown("""
        ## Chatbot  
        To chat, introduce a question and press enter.
                    
        Question examples:
                    
         - Hi
                    
         - What is the document about?
                    
         - What can visit in Lyon?                   
        """
        )
        # Chatbot
        chatbot = gr.Chatbot()
        
        # Input message
        msg = gr.Textbox(label="Question")
        
        # Clear button
        clear = gr.Button("Clear all (API key, document, chatbot))")

        # Init the LLM and read document with default parameters (if API key is provided)
        #if openai_api_key is not None:            
        #    init_read_doc(doc_type, doc_path, chunk_size, chunk_overlap, llm_name, temperature, openai_api_key)
        # -------------------------
        # When read document (aready read with default parameters)
        btn_read.click(reading_doc_msg,                                         # Reading message 
                            inputs=[drop_type, text_path], 
                            outputs=text_read_output).then(init_read_doc,   # Init qa chain and read document
                                inputs=[drop_type, text_path, 
                                        num_chunk_size, num_chunk_overlap,
                                        sl_temperature], 
                                queue=False).then(read_doc_msg,             # Finished reading message
                                        outputs=text_read_output).then(clear_chatbot_after_read_doc, # Clear chatbot
                                                outputs=[chatbot, msg], queue=False)  
        # -------------------------
        # When question 
        msg.submit(qa_input_msg_history, 
                     inputs=[msg, chatbot], 
                     outputs=[msg, chatbot], queue=False)#.then(bot, chatbot, chatbot)
        
        # When clear
        clear.click(clear_all, 
                    outputs=[text_openai_api_key_output, text_read_output, 
                             chatbot, msg, text_openai_api_key, text_path], queue=False)

        
    #demo.queue() # To use generator, required for streaming intermediate outputs
    demo.launch(share=share_gradio)


