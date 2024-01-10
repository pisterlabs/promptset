import os
import openai
from apikey import api_key
os.environ['OPENAI_API_KEY'] = api_key
import streamlit as st

openai.api_key = api_key  # Replace with your actual OpenAI API key

# Set page config
st.set_page_config(layout="centered")

st.title('Playwright GPT')
scene_dsc=st.text_input('Enter a place:')
char1_bio=st.text_input('Enter a character bio 1prompt:')
char2_bio=st.text_input('Enter a character bio 2prompt:')
plt_tws=st.text_input('Enter a Plot twist:')
total_lines=st.number_input('Enter total nbr of lines:' , min_value=0)
plt_tws_ln=st.number_input('After how many lines do you want the plot twist to occur? :',min_value=0)

generated_title=''

# char1_bio = "You are a military man"
# char2_bio = "You are a reporter"
# Start Play button
start_play = st.button('Start Play')
st.title(generated_title)
if start_play:
    char1_name = char1_bio.split()[0]  # Get the first word from the bio
    char2_name = char2_bio.split()[0]  # Get the first word from the bio

    # Define initial conversation history with character descriptions
    conversation_history1 = [
        {"role": "system", "content": f"You are in a play with only two characters. Maintain the format of a play. You will respond as a character from the play. You will only respond with one dialogue. you play {char1_name}: {char1_bio} and you are currently in {scene_dsc}"}
    ]
    conversation_history2 = [
        {"role": "system", "content": f"You are in a play with only two characters. Maintain the format of a play. You will respond as a character from the play. You will only respond with one dialogue. you play {char2_name}: {char2_bio} and you are currently in {scene_dsc}"}
    ]
    
    # Variables to store the dialogues in play format
    dialogue1 = ""

    def generate_response_character1(prompt):
        global conversation_history1
        global conversation_history2
        global dialogue1

        # Add the role's prompt to the conversation history
        conversation_history1.append({"role": "user", "content": f"{char2_name}: {prompt}"})
        # Use OpenAI's API to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history1,
            max_tokens=50
        )
        # The 'choices' field in the response contains the generated text
        generated_text = response['choices'][0]['message']['content']
        # Add the generated response to the conversation history
        conversation_history1.append({"role": "assistant", "content": f"{generated_text}"})
        conversation_history2.append({"role": "assistant", "content": f"{generated_text}"})
        # Add the dialogue to the play script
        dialogue1 += f"{generated_text}\n"
        st.markdown(generated_text)
        return generated_text
    def generate_response_character2(prompt):
        global conversation_history1
        global conversation_history2
        global dialogue1
 
        # Add the role's prompt to the conversation history
        conversation_history2.append({"role": "user", "content": f"{char1_name}: {prompt}"})
        # Use OpenAI's API to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history2,
            max_tokens=50
        )
        # The 'choices' field in the response contains the generated text
        generated_text = response['choices'][0]['message']['content']
        # Add the generated response to the conversation history
        conversation_history1.append({"role": "assistant", "content": f"{generated_text}"})
        conversation_history2.append({"role": "assistant", "content": f"{generated_text}"})
        # Add the dialogue to the play script
        dialogue1 += f"{generated_text}\n"
        st.markdown(generated_text)
        return generated_text
    def plot_twist(twist):
        global conversation_history1
        global conversation_history2
        global dialogue1
        # Add the plot twist to the conversation history
        conversation_history1.append({"role": "system", "content": twist})
        conversation_history2.append({"role": "system", "content": twist})
        # Add the plot twist to the play script
        dialogue1 += f"Narrator: {twist}\n"
        st.markdown(f"Narrator: {twist}")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history1,
            max_tokens=50
        )
        # The 'choices' field in the response contains the generated text
        generated_text = response['choices'][0]['message']['content']
    # Add the generated response to the conversation history
        conversation_history1.append({"role": "assistant", "content": f"{generated_text}"})
        conversation_history2.append({"role": "assistant", "content": f"{generated_text}"})
        # Add the dialogue to the play script
        dialogue1 += f"{generated_text}\n"
        st.markdown(generated_text)
    
        return generated_text
    
    #scene setting
    def generate_scene(scene_description):
        global conversation_history1
        global conversation_history2
        global dialogue1
        # Use OpenAI's API to generate a response
        response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=scene_description,
        temperature=0.5,
        max_tokens=200
        )
        generated_text = response.choices[0].text.strip()
        # Add the dialogue to the play script
        #dialogue1 += f"scene : {generated_text}\n"
        return generated_text
        

    #scene setting 
    scene_desc=f"describe the settings of a scene set in {scene_dsc}"
    response=generate_scene(scene_desc)
    st.subheader(response)

    # Initial prompt for Character 1
    initial_prompt = "Hello there!"
    # # Generate response for Character 1 and use it as input for Character 2
    # response1 = generate_response_character1(initial_prompt)
    # response2 = generate_response_character2(response1)


    # Generate responses for each character
    for i in range(plt_tws_ln):
        if i % 2 == 0:
            response = generate_response_character1(initial_prompt if i == 0 else response)
            
        else:
            response = generate_response_character2(response)
       
    # Introduce a plot twist
    #plot_twist = "Suddenly, a spaceship lands in the middle of the street. The people react to it"
    responsetwist = plot_twist(plt_tws)
    # Continue the conversation after the plot twist
    for i in range(total_lines-plt_tws_ln):
        if i % 2 == 0:
            response = generate_response_character1(responsetwist if i == 0 else response)
           
        else:
            response = generate_response_character2(response)
    
    # Generate a title for the play
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"genereate a title for the play below {dialogue1}",
        temperature=0.5,
        max_tokens=25
        )
    generated_title = response.choices[0].text.strip()
    
    st.header(generated_title)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"generate an intriguing teaser about the play {dialogue1}",
        temperature=0.7,
        max_tokens=100
        )
    generated_summary= response.choices[0].text.strip()
    
    st.caption(generated_summary)
    #st.markdown(f"```\n{dialogue1}\n```")