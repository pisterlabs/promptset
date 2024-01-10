import os
import streamlit as st
import openai
from openai import OpenAI
import tempfile
import random
import numpy as np

os.environ['OPENAI_API_KEY'] = st.secrets["key1"]
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],)

# App framework
st.title('🤖🍞  Talking toaster AI')
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

# Prompt template

def answer_template(answer):
        return f""" You are an experienced Electric Engineer specializing in household appliances or electronic equipment,
        your task is to assist individuals with no technical background in identifying and addressing technical issues.
        You will have access to the {answer}, grab this information as knowledge to reply.
        Maintain a helpful, friendly, clear, and concise tone throughout. Start by briefly describing the product {product_name} and confirming its equipment and model.
        Then, identify the issue and seek clarification with up to two simple, non-technical questions if needed. Provide a straightforward
        solution. Highlight common mispractices for the equipment. If the repair is too technical or potentially hazardous, advise seeking
        support from the equipment's brand or hiring a specialized technician. Your name is 'Talking Toaster',
        Make sure you only introduce yourself the first time. Say you going to take over the funny grandma to answer the questions also mention it only once.."use no more than 100 words."""

def answer_template_flow(template_flow):
        return f""" You are an experienced Electric Engineer specializing in household appliances or electronic equipment,
        your task is to assist individuals with no technical background in identifying and addressing technical issues.
        You will have access to the {answer} grab this information as knowledge to reply. Use the conversation_history to have context.
        Maintain a helpful, friendly, clear, and concise tone throughout.
        If the repair is too technical or potentially hazardous, advise seeking
        support from the equipment's brand or hiring a specialized technician. Your name is 'Talking Toaster',
        do not introduce yourself unless you are asked to."use no more than 100 words."""

template_flow = answer_template_flow

picture = st.camera_input("Take a picture", key="unique_picture_key")

# Save uploaded image to a temporary file
#product_name1 = None

#["Samsung Galaxy S23", "Toaster", "Microwave Oven", "Refrigerator", "Washing Machine", "Dishwasher"]
if picture:
    product_names = ["Dishwasher"]
    if 'product_name' not in st.session_state:
        st.session_state['product_name'] = random.choice(product_names)
    product_name = st.session_state['product_name']


    # Maintain conversation history

    if st.button(f"Product Name: {product_name}") or 'clicked' in st.session_state:
        st.session_state['clicked'] = True


        prompt_template = f"""first, will only anwser the first querie like You are a funny old lady always mad about household appliance malfunctions,
            acknowledge the {product_name} and saying something funny. you will finish the prompt saying,
            'How can i help you my dear?'. Use no more than 100 words."""



        if 'conversation_history' not in st.session_state:
            st.session_state["conversation_history"] = []
        conversation_history = st.session_state["conversation_history"]



        if 'velhinha' not in st.session_state:
            response = client.chat.completions.create(
                    messages=[{"role": "system", "content": prompt_template}],
                model="gpt-3.5-turbo", temperature= 0.8
            )
            #st.session_state['velhinha'] = response.choices[0].message.content

        #    st.text_area('Talking Toaster:', response.choices[0].message.content, height=100)
            st.session_state['velhinha'] = response.choices[0].message.content
            #conversation_history.append({"role":"assistant","content":st.session_state['velhinha']})

        st.write('Funny Grandma: ' + st.session_state['velhinha'])


##########################################################################################
##################################AUdio input############################################

        # Create speech synthesis
        response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",

            ### use the text from the input box###
            input= st.session_state['velhinha'],
        )

        # Save the audio to a file
        audio_file_path = "output.mp3"
        response.stream_to_file(audio_file_path)

        # Play the audio file
        st.audio(audio_file_path, format='audio/mp3')
##########################################################################################


        prompt = st.text_input('Ask the Toaster')
        if prompt:
            conversation_history.append({"role":"user","content": f"User: {prompt}"})

            #question : My dishwasher is making an unusual noise.

            answer = f"""DISHWASHER - NORMAL / ABNORMAL NOISES
There are many sounds you may hear when your dishwasher is in use. Some sounds are normal while others may be abnormal. Abnormal sounds sometimes have easy solutions to eliminate the sound. Here are some of the normal and abnormal sounds you may hear and suggestions to correct abnormal sounds.



The following is a list of Normal sounds:

Clicking or Grinding Sound in the Control: Dishwasher models with a Timer (Knob) control and some models with electronic controls will make clicking sounds as the dishwasher progresses through its cycle.

Some models of dishwashers with electronic controls have a motor driven electrical switching device in the control panel. This type of sound will be heard when the dishwasher is changing cycles; for example, when the dishwasher moves from circulation mode to pump-out mode. The sound is normal.

Humming: The dishwasher motor may make on and off humming sounds during operation that are normal. This sound occurs because:

The fan that cools the main pump motor is rotating.

The soft food disposer may be grinding up food waste.

The drain pump may be operating.

Pausing During Drain: The drain pump starts and stops several times during each drain. This helps pump out food debris more effectively. If your dishwasher pauses briefly or seems to start and stop during the draining period, this is normal operation.

Snapping Sound While Running:

Throughout the Cycle: A dishwasher fills and drains several times during the wash and rinse portions of the cycle. Each time it begins to drain a solenoid energizes which causes a definitive snapping sound. This will happen several times during each cycle and is dependent on the model and options selected. This is normal. Some models do not have drain solenoids, so this sound will not be heard on all models.

Loud Snap Mid-Cycle: It is normal to hear a snapping sound when the detergent cup opens.

Snapping When Opening the Door: It is normal to hear a snapping sound when the detergent cup opens as it hits the door or upper rack.

On flip cover detergent cups: The cover only opens to a 90-degree angle when in the wash cycle. When the door is opened at the end of the cycle, you may hear a snap as the cup opens fully. The sound is normal.

On some models: You may hear a snap from the door latch as you open and close the dishwasher. This is normal.

Swishing or Sloshing: The dishwasher will make a swishing noise as the water is sprayed around inside the tub during the wash and rinse portions of the cycle.

Whining Noise: The dishwasher wash pump motor can make a whining sound when operating. This is normal."""

            if 'Talking Toaster' not in st.session_state:

                conversation_history.append({"role":"system","content": answer_template(answer)})
                response = client.chat.completions.create(
                    messages=conversation_history,
                    model="gpt-3.5-turbo", temperature= 0.1
                )


                st.session_state['Talking Toaster'] = conversation_history.append({"role":"assistant","content": f"Talking Toaster: {response.choices[0].message.content}"})


            user_count = sum(1 for message in conversation_history if "User:" in message['content'])
            if user_count >= 2:
            # Do something...

                conversation_history.append({"role":"system","content": answer_template_flow(template_flow)})
                response = client.chat.completions.create(
                    messages=conversation_history,
                model="gpt-3.5-turbo", temperature= 0.1
            )
                conversation_history.append({"role":"assistant","content": response.choices[0].message.content})

        st.text("History")
        for i,value in enumerate(conversation_history):
            if value['role'] != 'system':
                st.write(value["content"])



else:
    if 'product_name' in st.session_state:
        del st.session_state["product_name"]
    if 'clicked' in st.session_state:
        del st.session_state['clicked']
    if 'conversation_history' in st.session_state:
        del st.session_state['conversation_history']
    if "velhinha" in st.session_state:
        del st.session_state['velhinha']
    if "Talking Toaster" in st.session_state:
        del st.session_state['Talking Toaster']
