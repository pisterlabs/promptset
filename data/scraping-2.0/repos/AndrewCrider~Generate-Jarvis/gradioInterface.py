import gradio as gr
import random
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
import openai

import elevenLabsGeneration
import apersonality.personalities as personas
import langchainBasic

response_location = "Generate-Jarvis/apersonality"
global current_personality
global personalities_encountered
current_personality = None
personalities_encountered = []

demo = gr.Blocks()
demo.queue()
def gradioApp():  
       
    with gr.Blocks() as demo:
       
        with gr.Row():
            
            with gr.Column(scale=1):
                audio = gr.Audio(autoplay= True, sources=['upload'], visible=True, streaming=False,)
                video = gr.Video(autoplay= True)
                image = gr.Image(visible=False)
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=800)
                msg = gr.Textbox()
                clear = gr.ClearButton([msg, chatbot])
        
            def respond(message, chat_history):
                global current_personality
                global personalities_encountered
                history_langchain_format = []
                audio_play = gr.Audio(render=False)
                
                personality_determiner = personas.PersonalityDeterminer(message)
                selected_personality = personality_determiner.determine_personalities()
                personality_name = selected_personality['personality']
                
                ## Add the Personality System Prompt
                history_langchain_format.append(SystemMessage(content=selected_personality['prompt']))
                for human, ai in chat_history:
                    history_langchain_format.append(HumanMessage(content=human))
                    history_langchain_format.append(AIMessage(content=ai))

                history_langchain_format.append(HumanMessage(content=message))
                gpt_response = langchainBasic.basicConversation(history_langchain_format)

                if (current_personality != personality_name ):
                    current_personality = selected_personality['personality']
                    
                    if personality_name not in personalities_encountered:
                        video_display = gr.Video(value=f"{response_location}{selected_personality['intro_video']}", visible=True)
                        personalities_encountered.append(personality_name)
                        static_image = gr.Image(visible=False)
                    
                    else:
                        video_display = gr.Video(render=False, visible=False)
                        static_image = gr.Image(value=f"{response_location}{selected_personality['static_image']}", visible=True)

                    
                else: 
                    video_display = gr.Video(visible=False)
                    static_image = gr.Image(value=f"{response_location}{selected_personality['static_image']}", visible=True)
                    if (len(gpt_response) < 1000):
                        audio_play = elevenLabsGeneration.getPersonalityVoice(selected_personality['voice'], gpt_response)

                print(history_langchain_format)
                print(gpt_response)
                
                chat_history.append((message, gpt_response))
                
                time.sleep(2)
                return "", chat_history, video_display, static_image, audio_play

        msg.submit(respond, [msg, chatbot], [msg, chatbot, video, image, audio])
        demo.launch(show_api=False, inbrowser=True)

    

           
if __name__ == "__main__":
    gradioApp()